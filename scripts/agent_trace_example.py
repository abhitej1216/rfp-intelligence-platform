from app.agents.orchestrator import Orchestrator
from app.agents.state import AgentState

# Example trace
state = AgentState()
orch = Orchestrator()
state = orch.plan("extract bid", state)
print("Plan:", state.plan)
