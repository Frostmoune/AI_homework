from pomegranate import *

# build the network
Burglary = DiscreteDistribution({'B': 0.001, '~B': 0.999})
Earthquake = DiscreteDistribution({'E': 0.002, '~E': 0.998})

Alarm = ConditionalProbabilityTable(
    [['B', 'E', 'A', 0.95],
    ['B', 'E', '~A', 0.05],
    ['B', '~E', 'A', 0.94],
    ['B', '~E', '~A', 0.06],
    ['~B', 'E', 'A', 0.29],
    ['~B', 'E', '~A', 0.71],
    ['~B', '~E', 'A', 0.001],
    ['~B', '~E', '~A', 0.999]], [Burglary, Earthquake])
JohnCalls = ConditionalProbabilityTable(
    [['A', 'J', 0.9],
    ['A', '~J', 0.1],
    ['~A', 'J', 0.05],
    ['~A', '~J', 0.95]], [Alarm])
MaryCalls = ConditionalProbabilityTable(
    [['A', 'M', 0.7],
    ['A', '~M', 0.3],
    ['~A', 'M', 0.01],
    ['~A', '~M', 0.99]], [Alarm])

S1 = Node(Burglary, name = 'Burglary')
S2 = Node(Earthquake, name = 'Earthquake')
S3 = Node(Alarm, name = 'Alarm')
S4 = Node(JohnCalls, name = 'JohnCalls')
S5 = Node(MaryCalls, name = 'MaryCalls')

model = BayesianNetwork("Burglary")
model.add_states(S1, S2, S3, S4, S5)
model.add_edge(S1, S3)
model.add_edge(S2, S3)
model.add_edge(S3, S4)
model.add_edge(S3, S5)
model.bake()

# P(A)
T1 = model.predict_proba({})[2].parameters[0]['A']
# P(A | J~M)
T3 = model.predict_proba({'JohnCalls':'J', 'MaryCalls':'~M'})[2].parameters[0]['A']
# P(A | B)
T4 = model.predict_proba({'Alarm':'A'})[0].parameters[0]['B']
# P(B | J~M)
T5 = model.predict_proba({'JohnCalls':'J', 'MaryCalls':'~M'})[0].parameters[0]['B']
# P(AJ~M)
P = model.probability([['B', 'E', 'A', 'J', '~M'],
                        ['~B', 'E', 'A', 'J', '~M'],
                        ['B', '~E', 'A', 'J', '~M'],
                        ['~B', '~E', 'A', 'J', '~M']]).sum()
# P(J~M) = P(AJ~M) / P(A | J~M)
T2 = P / T3
# P(~B)
P = model.predict_proba({})[0].parameters[0]['~B']
# P(J~M | ~B) = (1 - P(B | J~M)) * P(J~M) / P(~B)
T6 = (1 - T5) * T2 / P

print("P(A) = ")
print(T1)
print("\nP(J && ~M) = ")
print(T2)
print("\nP(A | J && ~M) = ")
print(T3)
print("\nP(B | A) = ")
print(T4)
print("\nP(B | J && ~M) = ")
print(T5)
print("\nP(J && ~M | ~B) = ")
print(T6)