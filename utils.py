import torch
import sklearn.linear_model
import sklearn.metrics as metrics

# privacy prediction
def evaluate_private_representations(encoder, train_dataset, test_dataset, device):
    encoder = encoder.to(device).eval()
    # train linear regression
    X_train, S_train = train_dataset.data, train_dataset.hidden
    z_train, _ = encoder(torch.FloatTensor(X_train).to(device))
    z_train = z_train.detach().cpu().numpy()

    s_predictor = sklearn.linear_model.LogisticRegression()
    s_predictor.fit(z_train, S_train)

    # test
    X_test, S_test = test_dataset.data, test_dataset.hidden
    z_test, _ = encoder(torch.FloatTensor(X_test).to(device))
    z_test = z_test.detach().cpu().numpy()

    s_pred_prob = s_predictor.predict_proba(z_test)
    # accuracy
    accuracy_s = metrics.accuracy_score(S_test, s_pred_prob)
    print(f'Accuracy on S (Logistic Regression): {accuracy_s}')

    return accuracy_s
