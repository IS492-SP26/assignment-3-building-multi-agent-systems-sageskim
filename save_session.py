"""
Run one query and save the full session as JSON + Markdown to outputs/
"""
import asyncio, json, os, yaml
from datetime import datetime
from pathlib import Path
os.chdir('/Users/mac/Desktop/assignment-3-building-multi-agent-systems-sageskim')

from dotenv import load_dotenv
load_dotenv('.env')
from src.autogen_orchestrator import AutoGenOrchestrator

with open('config.yaml') as f:
    config = yaml.safe_load(f)

orch = AutoGenOrchestrator(config)
query = "What are the latest trends in agentic UX design?"
print(f"Running query: {query}\n")

result = orch.process_query(query)

# Save JSON session
Path('outputs').mkdir(exist_ok=True)
session = {
    "timestamp": datetime.now().isoformat(),
    "query": result.get("query"),
    "response": result.get("response"),
    "conversation_history": result.get("conversation_history", []),
    "metadata": result.get("metadata", {}),
}
with open('outputs/example_session.json', 'w') as f:
    json.dump(session, f, indent=2)
print("Saved: outputs/example_session.json")

# Save Markdown report
md = f"""# Research Report

**Query:** {result.get('query')}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

{result.get('response', '')}
"""
with open('outputs/example_report.md', 'w') as f:
    f.write(md)
print("Saved: outputs/example_report.md")
print("\nDone!")
