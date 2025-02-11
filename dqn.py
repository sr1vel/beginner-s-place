import pandas as pd
original_df = pd.read_csv('Obfuscated-MalMem2022.csv', sep=',', encoding='utf- 8') original_df.shape
original_df.describe
()
original_df.tail(150)
df =
original_df.copy()
df.shape
import pandas as pd
def select_every_nth_row(df, n=1000):
selected_rows = df.iloc[::n]
return selected_rows
selected_rows = select_every_nth_row(df, n=1000) for column in df.columns:
if df[column].nunique() == 1:
print(f"All values in {column} are identical.")
columns_to_drop = ['pslist.nprocs64bit', 'handles.nport',
'svcscan.interactive_process_services'] df.drop(columns=columns_to_drop, inplace=True)
import pandas as pd
from scipy.stats import zscore
print("Number of Missing Values:")
print(df.isnull().sum())
print("\nNumber of Duplicate Rows:", df.duplicated().sum())
df.fillna(method="ffill", inplace=True) # Forward fill missing values
df.drop_duplicates(inplace=True)
df["Class"] = df["Class"].astype("category")
df = pd.get_dummies(df, columns=["Class"], drop_first=True)
print("\nData Info:")
print(df.info())
print("\nFirst Few
Rows:") print(df.head())
print("DataFrame Shape:", df.shape)
print("Class Distribution:", df["Class_Malware"].value_counts()) print("Number of
Missing Values:")
print(df.isnull().sum())
from sklearn.decomposition import
PCA import matplotlib.pyplot as plt
y = df["Class_Malware"]
X = df.drop(columns=["Category", "Class_Malware"]) from sklearn.preprocessing
import StandardScaler scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio =
pca.explained_variance_ratio_
31
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
plt.plot(cumulative_variance_ratio)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance
Ratio') plt.grid(True)
plt.show()
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components to retain for 95% variance: {n_components}")
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratios:", explained_variance_ratio) plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for the First 10 Principal Components')
plt.grid(True)
plt.show()
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
X_tsne = tsne.fit_transform(X_scaled)
print(X_tsne)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as
LDA lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)
X_lda
print(df["Class_Malware"].unique())
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label="Benign", color="blue", alpha=0.5)
axes[0].scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label="Malicious", color="red", alpha=0.5)
axes[0].set_title("PCA")
axes[0].legend()
axes[2].scatter(X_lda[y == 0, 0], X_lda[y == 0, 0], label="Benign", color="blue", alpha=0.5)
axes[2].scatter(X_lda[y == 1, 0], X_lda[y == 1, 0], label="Malicious", color="red", alpha=0.5)
axes[2].set_title("LDA")
axes[2].legend()
plt.show()
import numpy as np
import pandas as pd
import xgboost as
xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
feature_importances =
xgb_model.feature_importances_ N = 14
selected_feature_indices = np.argsort(feature_importances)[::- 1][:N] selected_features = X.columns[selected_feature_indices]
X_train_selected = X_train.iloc[:, selected_feature_indices]
X_test_selected = X_test.iloc[:, selected_feature_indices]
32
33
scaler = StandardScaler()
X_train_selected = scaler.fit_transform(X_train_selected)
X_test_selected = scaler.transform(X_test_selected)
N = 14 # Number of top features to select
sorted_indices = np.argsort(feature_importances)[::-1]
selected_features = X.columns[sorted_indices[:N]]
plt.figure(figsize=(10, 6))
plt.bar(selected_features, feature_importances[sorted_indices[:N]]) plt.title('Top Feature
Importance')
plt.xlabel('Features')
plt.ylabel('Importanc
e')
plt.xticks(rotation=4
5) plt.show()
selected_features
import numpy as np
import pandas as pd
import random
from collectionsimport deque
from tensorflow.keras.modelsimport Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
class DQNAgent:
def _init_(self, state_size, action_size): self.state_size =
state_size self.action_size =
action_size self.memory =
deque(maxlen=2000) self.gamma = 0.95 # discount
rate self.epsilon = 1.0 #
exploration rate self.epsilon_min = 0.01
self.epsilon_decay = 0.995
self.learning_rate = 0.001
self.model =
self._build_model()
def_build_model(self)
: model =
Sequential()
model.add(Dense(24, input_dim=self.state_size, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(self.action_size, activation='linear'))
model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate)) return model
def remember(self, state, action, reward, next_state, done):
self.memory.append((state, action, reward, next_state, done))
def act(self, state):
if np.random.rand() <= self.epsilon:
return random.randrange(self.action_size)
act_values = self.model.predict(state)
return np.argmax(act_values[0]) # returns
action def replay(self, batch_size):
minibatch = random.sample(self.memory,
batch_size) forstate, action, reward, next_state, done
in minibatch:
target =
reward if
not done:
target = (reward + self.gamma *
np.amax(self.model.predict(next_state)[0])
) target_f = self.model.predict(state)
target_f[0][action] = target
self.model.fit(state, target_f, epochs=1, verbose=0) if self.epsilon > self.epsilon_min:
self.epsilon *=
self.epsilon_decay def load(self, name):
self.model.load_weights(name)
def save(self, name):
self.model.save_weights(name)
defrandom_data_from_csv(file_path, features, num_samples=1000):
data = pd.read_csv(file_path)
selected_data = data[features].sample(n=num_samples)
return selected_data.values
deftrain_dqn(agent, data, episodes=1000, batch_size=32):
scores = []
for e in range(episodes):
state = data[np.random.randint(0, len(data))] # Sample
state state = np.reshape(state, [1, agent.state_size])
done =
False
score = 0
while not done:
action = agent.act(state)
next_state = data[np.random.randint(0, len(data))] # Sample next state
next_state = np.reshape(next_state, [1, agent.state_size])
reward = np.random.randn() # Sample reward
done = np.random.choice([True, False]) # Sample done
flag agent.remember(state, action, reward, next_state, done) state = next_state
score +=
reward if done:
break
if len(agent.memory) > batch_size:
agent.replay(batch_size)
scores.append(score)
print("Episode:", e+1, " Score:", score)
return scores
if _name_ == "_main_":
# Initialize environment
state_size = 14 # Number of selected features
action_size = 2 # Example action size
agent = DQNAgent(state_size, action_size)
features = ['svcscan.nservices', 'svcscan.process_services',
'handles.avg_handles_per_proc', 'handles.ndesktop',
'callbacks.ncallbacks', 'malfind.commitCharge',
'pslist.nproc', 'psxview.not_in_deskthrd',
'handles.nevent', 'pslist.avg_handlers',
'psxview.not_in_session_false_avg',
'malfind.protection', 'handles.nkey', 34
'ldrmodules.not_in_load']
selected_data = random_data_from_csv("Data Normalised.csv", features, num_samples=1000)
scores = train_dqn(agent, selected_data)
plt.plot(scores)
plt.xlabel('Episod
e')
plt.ylabel('Score')
plt.title('DQN Training
Performance') plt.show()
X_test = random_data_from_csv("Obfuscated-MalMem2022.csv", features, num_samples=100) y_test = np.random.randint(2, size=len(X_test)) # Example labels for
evaluation
y_pred = []
forstate in X_test:
state = np.reshape(state, [1, agent.state_size]) action = agent.act(state)
y_pred.append(action)
y_pred = np.array(y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion
Matrix') plt.show()
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)