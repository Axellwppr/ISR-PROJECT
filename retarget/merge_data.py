import joblib

data_a = joblib.load('/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/motions/xbot/ik_new_final_amass.pkl') # dict
data_b = joblib.load('/home/axell/desktop/11-30-ISR-PROJ/human2humanoid/resources/motions/xbot/ik_new_final_gen.pkl') # dict

data_ab = {**data_a, **data_b}

joblib.dump(data_ab, 'ik_new_final_amass_gen_origin.pkl')