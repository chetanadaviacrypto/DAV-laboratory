import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("uploads/titanicpreprocessed.csv", index_col=0)

total = len(df)
survived = int(df["Survived"].sum())
died = total - survived
survival_rate = survived / total * 100

gender_counts = df["Sex"].value_counts()

surv_sex = df.groupby("Sex")["Survived"].agg(["count", "sum", "mean"])
surv_sex["mean"] = (surv_sex["mean"] * 100).round(2)
gap_sex = surv_sex.loc["female", "mean"] - surv_sex.loc["male", "mean"]

age = df["Age"].dropna()

surv_class = df.groupby("Pclass")["Survived"].agg(["count", "sum", "mean"])
surv_class["mean"] = (surv_class["mean"] * 100).round(2)
gap_class = surv_class.loc[1, "mean"] - surv_class.loc[3, "mean"]

ct = df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean") * 100

print("Q1 Overall survival rate: %.2f%% (%d survived / %d total, %d died)" % (survival_rate, survived, total, died))
print("\nQ2 Gender distribution:")
print(gender_counts)
print("\nQ3 Survival by gender (%):")
print(surv_sex)
print("Gap (female - male): %.2f percentage points" % gap_sex)
print("\nQ4 Age distribution:")
print(age.describe().round(2))
print("\nQ5 Survival by class (%):")
print(surv_class)
print("Gap (1st - 3rd): %.2f percentage points" % gap_class)
print("\nSurvival % by class and gender:")
print(ct.round(2))

plt.style.use("seaborn-v0_8-whitegrid")
BLUE, ORANGE, GREEN = "#4C72B0", "#DD8452", "#55A868"

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Titanic Passenger Analysis", fontsize=16, fontweight="bold")

ax = axes[0, 0]
bars = ax.bar(["Died", "Survived"], [died, survived], color=[ORANGE, GREEN], width=0.55)
ax.set_title("Q1. Overall Survival - %.1f%% survived" % survival_rate)
ax.set_ylabel("Passengers")
ax.set_ylim(0, max(died, survived) * 1.18)
for b, v in zip(bars, [died, survived]):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, "%d\n(%.1f%%)" % (v, v / total * 100),
            ha="center", fontsize=10)

ax = axes[0, 1]
vals = gender_counts.values
labels = gender_counts.index
ax.pie(vals, labels=["%s\n%d (%.1f%%)" % (l, v, v / total * 100) for l, v in zip(labels, vals)],
       colors=[BLUE, ORANGE], startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
ax.set_title("Q2. Gender Distribution")

ax = axes[1, 0]
order = ["female", "male"]
bars = ax.bar(order, [surv_sex.loc[g, "mean"] for g in order], color=[ORANGE, BLUE], width=0.55)
ax.set_title("Q3. Survival Rate by Gender (gap %.1f pp)" % gap_sex)
ax.set_ylabel("Survival rate (%)")
ax.set_ylim(0, 100)
for b, g in zip(bars, order):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
            "%.1f%%\n(%d/%d)" % (b.get_height(), surv_sex.loc[g, "sum"], surv_sex.loc[g, "count"]),
            ha="center", fontsize=10)

ax = axes[1, 1]
ax.hist(age, bins=range(0, 85, 5), color=BLUE, edgecolor="white")
ax.axvline(age.mean(), color="red", linestyle="--", label="Mean = %.1f yrs" % age.mean())
ax.axvline(age.median(), color="green", linestyle="--", label="Median = %.1f yrs" % age.median())
ax.set_title("Q4. Age Distribution")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Passengers")
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("titanic_charts_1.png", dpi=130)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes2[0]
class_labels = ["1st", "2nd", "3rd"]
bars = ax.bar(class_labels, [surv_class.loc[i, "mean"] for i in [1, 2, 3]],
              color=[GREEN, BLUE, ORANGE], width=0.55)
ax.set_title("Q5. Survival Rate by Class (1st-3rd gap %.1f pp)" % gap_class)
ax.set_ylabel("Survival rate (%)")
ax.set_ylim(0, 100)
for b, i in zip(bars, [1, 2, 3]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
            "%.1f%%\n(%d/%d)" % (b.get_height(), surv_class.loc[i, "sum"], surv_class.loc[i, "count"]),
            ha="center", fontsize=10)

ax = axes2[1]
x = [0, 1, 2]
w = 0.38
ax.bar([i - w / 2 for i in x], [ct.loc[i, "female"] for i in [1, 2, 3]], w, label="Female", color=ORANGE)
ax.bar([i + w / 2 for i in x], [ct.loc[i, "male"] for i in [1, 2, 3]], w, label="Male", color=BLUE)
ax.set_xticks(x, class_labels)
ax.set_title("Survival % by Class & Gender")
ax.set_ylabel("Survival rate (%)")
ax.legend()
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("titanic_charts_2.png", dpi=130)
