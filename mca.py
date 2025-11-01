import pandas as pd
import prince

# Mixed dataset
df = pd.DataFrame({
    'Age': [22, 35, 28, 41, 30],
    'Income': [2500, 4000, 3000, 4500, 3200]
})

# Apply FAMD (combines PCA + MCA)
mca = prince.MCA(n_components=2, random_state=42)
mca = mca.fit(df)
df_mca = mca.transform(df)

print(df_mca)
print("total inertia:", mca.total_inertia_)
