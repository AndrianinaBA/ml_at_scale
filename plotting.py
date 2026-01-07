import matplotlib.pyplot as plt
import os


directory = "./"
elements = os.listdir(directory)

results = [elt for elt in elements if elt.startswith("result") and elt.endswith(".txt")]
print(results)
print(len(results))

for file in results:
    filename = file.split(sep="_")
    last_three = filename[-3:]
    lambda_ = float(last_three[0])
    gamma_ = float(last_three[1])
    tau_ = float(str(last_three[2][:3]))
    tracker = []
    rmse = []
    with open(file, "r") as fp:
        for line in fp:
            tracker_rmse = line.strip()
            splitted = tracker_rmse.split(sep='\t') 
            print(splitted)
            tracker.append(float(splitted[0]))
            rmse.append(float(splitted[1]))
    
    if rmse[-2] >= rmse[-1]: # Only those who are strictly decreasing
        plt.plot(tracker, rmse)
        plt.title(f"λ = {lambda_}  γ = {gamma_} τ = {tau_}")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.savefig(f"{file}_re.pdf")
        plt.clf()

    