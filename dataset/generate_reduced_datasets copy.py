import pandas as pd



toniot_dist = [
    ["xss", 48652],
    ["Benign", 54024],
    ["scanning", 21335],
    ["ddos", 40000],
    ["dos", 12665],
    ["injection", 13686],
    ["password", 22557],
    ["backdoor", 352],
    ["mitm", 124],
    ["ransomware", 65]
]

botiot_dist = [
    ["DDoS", 184552],
    ["DoS", 167004],
    ["Reconnaissance", 25907],
    ["Benign", 1285],
    ["Theft", 33]
]

nb15_dist = [
    ["Benign", 455751],
    ["Exploits", 5902],
    ["Reconnaissance", 1309],
    ["Fuzzers", 3039],
    ["Backdoor", 65],
    ["Analysis", 71],
    ["Generic", 576],
    ["Shellcode", 127],
    ["DoS", 661],
    ["Worms", 38]
]

cicids_dist = [
    ["Benign", 154649],
    ["DDoS", 13828],
    ["Infilteration", 1087],
    ["DoS", 4855],
    ["Brute Force", 1202],
    ["Bot", 1510],
    ["SQL Injection", 8]
]


toniot = pd.read_csv("/data176/privatecloud/data/autodl-container-1b164bbe69-af1a6422-storage/ids_data/NF-ToN-IoT-v2.csv")
print("> TON-IoT")
for attack in toniot_dist:
    num_samples_attack = toniot[toniot['Attack'] == attack[0]].shape[0]
    q = num_samples_attack - attack[1]
    print(q)
    print(">> Sampling TON-IoT ({}) - Num of samples {} | Required Samples {} | To be removed {}".format(attack[0], num_samples_attack, attack[1], q))
    toniot = toniot.drop(toniot[toniot['Attack'] == attack[0]].sample(n=q).index)
toniot.to_csv("/data176/privatecloud/data/autodl-container-1b164bbe69-af1a6422-storage/reduced_ids_data/toniot_bench_dnn.csv", index=False)


botiot = pd.read_csv("/data176/privatecloud/data/autodl-container-1b164bbe69-af1a6422-storage/ids_data/NF-BoT-IoT-v2.csv")
print("> Bot-IoT")
for attack in botiot_dist:
    num_samples_attack = botiot[botiot['Attack'] == attack[0]].shape[0]
    q = num_samples_attack - attack[1]
    print(">> Sampling Bot-IoT ({}) - Num of samples {} | Required Samples {} | To be removed {}".format(attack[0], num_samples_attack, attack[1], q))
    botiot = botiot.drop(botiot[botiot['Attack'] == attack[0]].sample(n=q).index)
botiot.to_csv("/data176/privatecloud/data/autodl-container-1b164bbe69-af1a6422-storage/reduced_ids_data/botiot_bench_dnn.csv", index=False)


nb15 = pd.read_csv("/data176/privatecloud/data/autodl-container-1b164bbe69-af1a6422-storage/ids_data/NF-UNSW-NB15-v2.csv")
print("> UNSW-NB15")
for attack in nb15_dist:
    num_samples_attack = nb15[nb15['Attack'] == attack[0]].shape[0]
    q = num_samples_attack - attack[1]
    print(">> Sampling UNSW-NB15 ({}) - Num of samples {} | Required Samples {} | To be removed {}".format(attack[0], num_samples_attack, attack[1], q))
    nb15 = nb15.drop(nb15[nb15['Attack'] == attack[0]].sample(n=q).index)
nb15.to_csv("/data176/privatecloud/data/autodl-container-1b164bbe69-af1a6422-storage/reduced_ids_data/nb15_bench_dnn.csv", index=False)



cicids = pd.read_csv("/data176/privatecloud/data/autodl-container-1b164bbe69-af1a6422-storage/ids_data/NF-CSE-CIC-IDS2018-v2.csv")
##处理NF-CSE-CIC-IDS2018-v2.csv,将细分的共计类别聚集到DoS,DDoS,Brute Force              
ddos=cicids[cicids['Attack'].str.contains("ddos",case=False)]
dos=cicids[cicids['Attack'].str.contains("dos",case=False)]
cicids.drop(index=dos.index,inplace=True)
dos.drop(index=ddos.index,inplace=True)

brute=cicids[cicids['Attack'].str.contains("brute",case=False)]
cicids.drop(index=brute.index,inplace=True)
print(cicids.Attack.value_counts())
print("before subsampling")
print(dos.Attack.value_counts())
print(ddos.Attack.value_counts())
print(brute.Attack.value_counts())

dos=pd.concat(objs=[dos])

ddos=pd.concat(objs=[ddos])

brute=pd.concat(objs=[brute])


print("after subsampling")
print(dos.Attack.value_counts())
print(ddos.Attack.value_counts())
print(brute.Attack.value_counts())

dos.Attack="DoS"
ddos.Attack="DDoS"
brute.Attack="Brute Force"
cicids=pd.concat(objs=[cicids, dos,ddos,brute])
print("> CIC-IDS-2018")
for attack in cicids_dist:
    num_samples_attack = cicids[cicids['Attack'] == attack[0]].shape[0]
    q = num_samples_attack - attack[1]
    print(">> Sampling CIC-IDS-2018 ({}) - Num of samples {} | Required Samples {} | To be removed {}".format(attack[0], num_samples_attack, attack[1], q))
    cicids = cicids.drop(cicids[cicids['Attack'] == attack[0]].sample(n=q).index)
cicids.to_csv("/data176/privatecloud/data/autodl-container-1b164bbe69-af1a6422-storage/reduced_ids_data/cicids_bench_dnn.csv", index=False)
