# v2固定竞态调度

`race-schedules.json`固定三类各100条输入/屏障调度组合。seed是可重放标识，测试以JSON中的实际参数执行，不依赖运行时随机sleep。

```powershell
python -m pytest tests/test_v2_race_schedules.py -q
python -m pytest tests/test_v2_race_schedules.py -q -k 'cancel_commit-310000'
```

重放时以清单中已有seed为准；失败输出包含完整调度参数。修改清单需同时通过覆盖检查，不把重复的相同条件计作新增调度。

| 字段 | 含义 |
| --- | --- |
| family | cancel_commit、takeover_commit或ack_interrupt |
| phase | 明确的屏障停靠位置 |
| multi | 是否先提交一个搜索动作，再测试详情动作 |
| paid_calls | 目标动作内模拟Provider调用数，1—3次 |
| refreshes | 中断前读取ORM运行记录的次数，检查缓存视图不代替SQL裁决 |
| duplicates | 每次结算后重复提交回执的次数，重复请求不得改写费用 |
| worker | 模型请求在Action线程还是继承上下文的Worker中执行 |
| interrupt | 取消或模拟租约结束后由真实acquire领取新执行 |

提交组覆盖业务结果已计算、CAS前、CAS已持写锁、提交后。CAS持锁时，测试先确认另一写入已进入SQL，再放行原提交，避免把线程启动时间当作提交顺序。

ACK组覆盖工具确认前后、费用预留确认后、Provider内部及结算后。已知返回精确对账；接管时未知结果对应持久预留及uncertain，迟到响应不能改写新Owner账本。

SQLite、ActionExecutor、Controller、费用与恢复路径使用实际实现，Provider返回使用离线替身。本清单覆盖有限的控制窗口及负载组合，不用于估算生产故障概率、吞吐或模型质量。真实进程强退另见`test_v2_process_recovery.py`与`test_v2_section_recovery.py`。

## 冻结业务输入

`frozen-business-input.json`是明确标注的合成材料，供`tests/test_v2_frozen_business.py`验证四角色、两种协议和三种局部策略。记录标识只测试引用格式，不代表存在真实在线论文；脚本蕴含标签不作为人工金标或模型准确率。

```powershell
python -m pytest tests/test_v2_frozen_business.py tests/test_v2_business_flows.py -q
```

测试保留实际renderer、引用/授权一致性、Check和最终门禁，同时覆盖未核验结束、引用不足、旧报告失效、矛盾、澄清恢复、发布拒绝、中断与草稿保留。它属于正确性回归，不进入策略效果统计。
