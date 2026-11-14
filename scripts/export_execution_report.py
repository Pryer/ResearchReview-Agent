"""只读回放历史E2E目录，使用项目正式报告器导出HTML。"""
from pathlib import Path
import argparse
import json
import os
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True,help='含research.db的已结束运行目录；相对路径以项目根目录解析')
    parser.add_argument('--session-id',help='默认读取run.json中的run_id或目录名')
    parser.add_argument('--output-dir',type=Path,help='默认使用项目配置的报告目录；相对路径以项目根目录解析')
    args=parser.parse_args(argv)
    # CLI独立进程在项目根目录加载配置，避免从其它终端目录读取错误.env。
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT))
    from sqlalchemy import create_engine,select,func
    from sqlalchemy.orm import Session
    from app.database.models import ResearchRuntimeModel,ResearchCheckpointModel
    from app.services.execution_report_capture import Capture
    from app.services.execution_report_service import freeze_source,build_report,write_report,report_directory,report_id
    run=args.run_dir if args.run_dir.is_absolute() else ROOT/args.run_dir
    database=run/'research.db'
    if not database.is_file():parser.error('运行目录缺少research.db')
    manifest=json.loads((run/'run.json').read_text(encoding='utf-8')) if (run/'run.json').is_file() else {}
    if manifest.get('task_status') in {'running','queued','cancel_requested'}:parser.error('任务尚未结束，请结束后再回放')
    sid=args.session_id or manifest.get('run_id') or run.name
    engine=create_engine('sqlite://',creator=lambda:sqlite3.connect('file:'+database.resolve().as_posix()+'?mode=ro',uri=True))
    try:
        with Session(engine) as db:
            runtime=db.get(ResearchRuntimeModel,sid)
            if runtime is None:parser.error('数据库中不存在指定会话')
            if runtime.owner is not None:parser.error('该会话仍有执行租约，请结束后再回放')
            request=(manifest.get('task') or {}).get('request') or {}
            state=json.loads(runtime.state_json or '{}')
            cap=Capture(sid,request.get('user_query') or state.get('user_query',''),sid)
            cap.start_version=0
            cap.end_version=db.scalar(select(func.max(ResearchCheckpointModel.version)).where(ResearchCheckpointModel.session_id==sid)) or 0
            cap.source=freeze_source(db,cap)
        result=json.loads((run/'result.json').read_text(encoding='utf-8')) if (run/'result.json').is_file() else {'status':manifest.get('task_status') or 'unknown'}
        report=build_report(cap,result)
        if isinstance(manifest.get('latency_seconds'),(int,float)):
            report['summary']['seconds']=manifest['latency_seconds'];report['meta']['seconds']=manifest['latency_seconds']
        output=args.output_dir or report_directory()/report_id(sid)
        if not output.is_absolute():output=ROOT/output
        write_report(report,output)
        print(str((output/'report.html').resolve()))
        return 0
    finally:engine.dispose()


if __name__=='__main__':raise SystemExit(main())
