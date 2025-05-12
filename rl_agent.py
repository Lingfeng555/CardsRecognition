from rl.decision_maker import DecisionMaker

decision_maker = DecisionMaker()
state = (0, 7, 0)
action = decision_maker.decide(state)
print(f"State: {state}, Action: {'Stand' if action == 0 else 'Hit'}")