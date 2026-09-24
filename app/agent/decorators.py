"""节点装饰器 - 显式声明依赖和输出。

使用方式：
    from app.agent.decorators import node, requires, provides
    
    @node(name="plan")
    @requires("user_query")
    @provides("intent", "topic", "keywords", "start_year", "end_year")
    def plan_node(state):
        # 节点逻辑
        pass
    
    @node(name="search")
    @requires("topic", "keywords", "start_year", "end_year")
    @provides("candidate_papers")
    def search_node(state):
        # 节点逻辑
        pass

功能：
1. 显式声明依赖：明确节点需要哪些输入
2. 显式声明输出：明确节点产生哪些输出
3. 自动验证：运行前检查依赖是否满足，运行后检查声明的输出是否产生
4. 注册表查询：``get_node_metadata`` 供契约测试核对节点声明
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Optional, Set

from app.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 节点元数据
# ============================================================

class NodeMetadata:
    """节点元数据"""
    
    def __init__(self, name: str):
        self.name = name
        self.required_fields: Set[str] = set()
        self.provided_fields: Set[str] = set()
        self.optional_fields: Set[str] = set()
        self.description: str = ""
        self.category: str = "general"
    
    def __repr__(self) -> str:
        return (
            f"NodeMetadata(name={self.name}, "
            f"requires={self.required_fields}, "
            f"provides={self.provided_fields})"
        )


# 全局节点注册表
_NODE_REGISTRY: Dict[str, NodeMetadata] = {}


def register_node(metadata: NodeMetadata) -> None:
    """注册节点元数据"""
    _NODE_REGISTRY[metadata.name] = metadata


def get_node_metadata(name: str) -> Optional[NodeMetadata]:
    """获取节点元数据"""
    return _NODE_REGISTRY.get(name)


# ============================================================
# 装饰器
# ============================================================

def node(
    name: str,
    description: str = "",
    category: str = "general",
):
    """节点装饰器
    
    Args:
        name: 节点名称
        description: 节点描述
        category: 节点类别 (planning / retrieval / generation / execution)
    """
    def decorator(func: Callable) -> Callable:
        # 创建或获取元数据。
        # 注意：@requires / @provides / @optional 通常写在 @node 下方，
        # Python 装饰器自底向上应用，因此它们会先执行并已经创建了
        # _node_metadata（此时用的是函数名做占位）。这里必须用 @node 传入的
        # name/description/category 覆盖占位值，并且无论元数据是否已存在，
        # 都要执行 register_node，否则节点永远不会被注册到全局表中。
        if not hasattr(func, "_node_metadata"):
            func._node_metadata = NodeMetadata(name)
        metadata = func._node_metadata
        metadata.name = name
        metadata.description = description or func.__doc__ or ""
        metadata.category = category
        register_node(metadata)
        
        @functools.wraps(func)
        def wrapper(state, *args, **kwargs):
            metadata = func._node_metadata
            
            # 执行前验证
            missing = validate_requirements(state, metadata)
            if missing:
                # 一些节点有受支持的降级输入（例如 retrieval_target 缺失时用
                # max_papers）。保留执行能力，但把契约违例写入状态供审计，
                # 不再只是日志里的一条无结构警告。
                state.setdefault("contract_violations", []).append({
                    "node": metadata.name,
                    "missing_fields": sorted(missing),
                })
                logger.warning(
                    "Node %s missing required fields: %s",
                    metadata.name,
                    missing,
                )
            
            # 执行节点
            t0 = time.time()
            try:
                result = func(state, *args, **kwargs)
                duration_ms = int((time.time() - t0) * 1000)
                
                # 验证输出
                provided = verify_provides(state, metadata)
                if provided < len(metadata.provided_fields):
                    logger.warning(
                        "Node %s only provided %d/%d expected fields",
                        metadata.name,
                        provided,
                        len(metadata.provided_fields)
                    )
                
                logger.debug(
                    "Node %s completed in %dms",
                    metadata.name,
                    duration_ms
                )
                
                return result
            except Exception as e:
                logger.error(
                    "Node %s failed: %s",
                    metadata.name,
                    str(e)
                )
                raise
        
        return wrapper
    return decorator


def requires(*fields: str):
    """声明节点需要的字段
    
    Args:
        *fields: 必需的状态字段名
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, "_node_metadata"):
            func._node_metadata = NodeMetadata(func.__name__)
        func._node_metadata.required_fields.update(fields)
        return func
    return decorator


def provides(*fields: str):
    """声明节点产生的字段
    
    Args:
        *fields: 节点会设置的状态字段名
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, "_node_metadata"):
            func._node_metadata = NodeMetadata(func.__name__)
        func._node_metadata.provided_fields.update(fields)
        return func
    return decorator


def optional(*fields: str):
    """声明节点可选的字段
    
    Args:
        *fields: 可选的状态字段名（有更好，没有也行）
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, "_node_metadata"):
            func._node_metadata = NodeMetadata(func.__name__)
        func._node_metadata.optional_fields.update(fields)
        return func
    return decorator


# ============================================================
# 验证函数
# ============================================================

def validate_requirements(
    state: Dict[str, Any],
    metadata: NodeMetadata
) -> List[str]:
    """验证节点依赖是否满足
    
    Returns:
        缺失的字段列表
    """
    missing = []
    for field in metadata.required_fields:
        # 支持嵌套字段（如 "planning.topic"）
        if "." in field:
            parts = field.split(".")
            current = state
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    missing.append(field)
                    break
        else:
            if field not in state or state.get(field) is None:
                missing.append(field)
    
    return missing


def verify_provides(
    state: Dict[str, Any],
    metadata: NodeMetadata
) -> int:
    """验证节点是否产生了声明的输出
    
    Returns:
        实际产生的字段数量
    """
    count = 0
    for field in metadata.provided_fields:
        if "." in field:
            parts = field.split(".")
            current = state
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    break
            else:
                count += 1
        else:
            if field in state and state.get(field) is not None:
                count += 1
    
    return count
