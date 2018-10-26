from pomegranate import *

# build the network
PatientAge = DiscreteDistribution({'0-30': 0.10, '31-65': 0.30, '65+': 0.60})
CTScanResult = DiscreteDistribution({'Ischemic Stroke': 0.7, 'Hemmorraghic Stroke': 0.3})
MRIScanResult = DiscreteDistribution({'Ischemic Stroke': 0.7, 'Hemmorraghic Stroke': 0.3})
Anticoagulants = DiscreteDistribution({'Used': 0.5, 'Not used': 0.5})

StrokeType = ConditionalProbabilityTable([
    ['Ischemic Stroke', 'Ischemic Stroke', 'Ischemic Stroke', 0.8], 
    ['Ischemic Stroke', 'Hemmorraghic Stroke', 'Ischemic Stroke', 0.5],   
    ['Hemmorraghic Stroke', 'Ischemic Stroke', 'Ischemic Stroke', 0.5], 
    ['Hemmorraghic Stroke', 'Hemmorraghic Stroke', 'Ischemic Stroke', 0],  

    ['Ischemic Stroke', 'Ischemic Stroke', 'Hemmorraghic Stroke', 0], 
    ['Ischemic Stroke', 'Hemmorraghic Stroke', 'Hemmorraghic Stroke', 0.4],  
    ['Hemmorraghic Stroke', 'Ischemic Stroke', 'Hemmorraghic Stroke', 0.4], 
    ['Hemmorraghic Stroke', 'Hemmorraghic Stroke', 'Hemmorraghic Stroke', 0.9], 

    ['Ischemic Stroke', 'Ischemic Stroke', 'Stroke Mimic', 0.2], 
    ['Ischemic Stroke', 'Hemmorraghic Stroke', 'Stroke Mimic', 0.1],     
    ['Hemmorraghic Stroke', 'Ischemic Stroke', 'Stroke Mimic', 0.1], 
    ['Hemmorraghic Stroke', 'Hemmorraghic Stroke', 'Stroke Mimic', 0.1]]
, [CTScanResult, MRIScanResult])

Mortality = ConditionalProbabilityTable([
    ['Ischemic Stroke', 'Used', 'False', 0.28],
    ['Hemmorraghic Stroke', 'Used', 'False', 0.99],
    ['Stroke Mimic', 'Used', 'False', 0.1],
    ['Ischemic Stroke', 'Not used', 'False', 0.56],
    ['Hemmorraghic Stroke', 'Not used', 'False', 0.58],
    ['Stroke Mimic', 'Not used', 'False', 0.05],

    ['Ischemic Stroke',  'Used' ,'True', 0.72],
    ['Hemmorraghic Stroke', 'Used', 'True', 0.01],
    ['Stroke Mimic', 'Used', 'True', 0.9],
    ['Ischemic Stroke',  'Not used', 'True', 0.44],
    ['Hemmorraghic Stroke', 'Not used', 'True', 0.42],
    ['Stroke Mimic', 'Not used', 'True', 0.95]
], [StrokeType, Anticoagulants])

Disability = ConditionalProbabilityTable([
    ['Ischemic Stroke',   '0-30','Negligible', 0.80],
    ['Hemmorraghic Stroke', '0-30','Negligible', 0.70],
    ['Stroke Mimic', '0-30', 'Negligible', 0.9],
    ['Ischemic Stroke', '31-65', 'Negligible', 0.60],
    ['Hemmorraghic Stroke', '31-65', 'Negligible', 0.50],
    ['Stroke Mimic', '31-65', 'Negligible', 0.4],
    ['Ischemic Stroke', '65+', 'Negligible', 0.30],
    ['Hemmorraghic Stroke', '65+', 'Negligible', 0.20],
    ['Stroke Mimic', '65+', 'Negligible', 0.1],

    ['Ischemic Stroke', '0-30', 'Moderate', 0.1],
    ['Hemmorraghic Stroke', '0-30', 'Moderate', 0.2],
    ['Stroke Mimic', '0-30', 'Moderate', 0.05],
    ['Ischemic Stroke', '31-65', 'Moderate', 0.3],
    ['Hemmorraghic Stroke', '31-65', 'Moderate', 0.4],
    ['Stroke Mimic', '31-65', 'Moderate', 0.3],
    ['Ischemic Stroke',  '65+', 'Moderate', 0.4],
    ['Hemmorraghic Stroke', '65+', 'Moderate', 0.2],
    ['Stroke Mimic', '65+', 'Moderate', 0.1],

    ['Ischemic Stroke', '0-30', 'Severe', 0.1],
    ['Hemmorraghic Stroke', '0-30', 'Severe', 0.1],
    ['Stroke Mimic', '0-30', 'Severe', 0.05],
    ['Ischemic Stroke', '31-65', 'Severe', 0.1],
    ['Hemmorraghic Stroke', '31-65', 'Severe', 0.1],
    ['Stroke Mimic', '31-65', 'Severe', 0.3],
    ['Ischemic Stroke', '65+', 'Severe', 0.3],
    ['Hemmorraghic Stroke', '65+', 'Severe', 0.6],
    ['Stroke Mimic', '65+', 'Severe', 0.8]
], [StrokeType, PatientAge])

S1 = Node(PatientAge, name = 'PatientAge')
S2 = Node(CTScanResult, name = 'CTScanResult')
S3 = Node(MRIScanResult, name = 'MRIScanResult')
S4 = Node(StrokeType, name = 'StrokeType')
S5 = Node(Anticoagulants, name = 'Anticoagulants')
S6 = Node(Mortality, name = 'Mortality')
S7 = Node(Disability, name = 'Disability')

model = BayesianNetwork("Burglary")
model.add_states(S1, S2, S3, S4, S5, S6, S7)
model.add_edge(S2, S4)
model.add_edge(S3, S4)
model.add_edge(S4, S6)
model.add_edge(S5, S6)
model.add_edge(S1, S7)
model.add_edge(S4, S7)
model.bake()

P1 = model.predict_proba({'PatientAge': '31-65', 
    'CTScanResult': 'Ischemic Stroke'})[5].parameters[0]['True']
print("P1 = ", P1)

P2 = model.predict_proba({'PatientAge':'65+', 
    'MRIScanResult': 'Hemmorraghic Stroke'})[6].parameters[0]['Moderate']
print("P2 = ", P2)

P3 = model.predict_proba({'PatientAge':'65+', 
    'CTScanResult': 'Hemmorraghic Stroke', 
    'MRIScanResult': 'Ischemic Stroke'})[3].parameters[0]['Stroke Mimic']
print("P3 = ", P3)

P4 = model.predict_proba({'PatientAge': '0-30'})[4].parameters[0]['Not used']
print("P4 = ", P4)