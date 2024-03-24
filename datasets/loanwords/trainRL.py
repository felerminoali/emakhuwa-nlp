from dataload import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

print('Train RL')
display_fields = display + features

f1_scores = []
precision_scores = []
recall_scores = []
accuracy_scores = []

y_test = []
y_pred = []

for i in range(5):
	state = random.randint(0, 42)
	print(state)

	train_set = train_alldata[display_fields + labels]
	train_set = train_set.sample(frac=1, random_state=state)
	x_train = train_set[features].values
	x_means = np.mean(x_train, axis=0)
	x_stds = np.std(x_train, axis=0)
	x_stds[x_stds == 0] = 1
	y_train = train_set[labels].values.ravel()

	test_set = test_alldata[display_fields + labels]
	test_set =  test_set.sample(frac=1, random_state=state)
	x_test = test_set[features].values
	y_test = test_set[labels].values.ravel()

	x_train = (x_train - x_means)/x_stds

	LR = LogisticRegression(random_state=1, solver='lbfgs', multi_class='ovr', max_iter=500).fit(x_train, y_train)
		
	print("Evaluating on all langs")
	x_test = (x_test - x_means)/x_stds
	y_pred = LR.predict(x_test)

	f1_scores.append(f1_score(y_test, y_pred ))
	precision_scores.append(precision_score(y_test, y_pred))
	recall_scores.append(recall_score(y_test, y_pred ))
	accuracy_scores.append(accuracy_score(y_test, y_pred))

repot_txt = f"""
f1-score :  { np.mean(f1_scores)} { np.std(f1_scores)}
precision :  {np.mean(precision_scores)} {np.std(precision_scores)}
recall :  {np.mean(recall_scores)} {np.std(recall_scores)}
accuracy :  {np.mean(accuracy_scores)} {np.std(accuracy_scores)}
"""
print(repot_txt)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
class_report = classification_report(y_test, y_pred)

save_predition(test_set, y_test, y_pred, split+'-RL', class_report, repot_txt)
