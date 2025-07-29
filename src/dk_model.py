import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DeepKriging(nn.Module):
    def __init__(self, input_size):
        super(DeepKriging, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(100),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.model(x)

class DeepKrigingTrainer:
    def __init__(self, deposit_data, covariates=None, regular_nn=False, plot_errors=True):
        self.deposit_data = deposit_data
        self.covariates = covariates
        self.phi_columns = self.deposit_data.columns[10:].tolist()
        if regular_nn:
            self.phi_columns = ['X', 'Y', 'Z']
        if self.covariates is not None:
            self.p = len(self.covariates) + len(self.phi_columns)
        else:
            self.p = len(self.phi_columns)
        self.plot_errors = plot_errors
        self.test_mse_list = []
        self.test_mae_list = []
        self.test_adjusted_r2_list = []
        self.test_r2_list = []
        torch.manual_seed(42)
        np.random.seed(42)


    def train_model(self, model, optimizer, criterion):
        self.train_losses = []  
        self.test_losses = []   
        # Training loop
        #for step in range(2001): #For Shapley
        for step in range(4601):
            model.train()
            pre = model(self.x_train)
            mse = criterion(pre, self.y_train)
            cost = mse

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            self.train_losses.append(mse.item())

        #     pre_test = model(self.x_test)
        #     mse_test = criterion(pre_test, self.y_test)
        #     self.test_losses.append(mse_test.item())


        # if self.plot_errors:
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(self.train_losses[150:], label='Train MSE')
        #     plt.plot(self.test_losses[150:], label='Test MSE')
        #     plt.xlabel('Epoch')
        #     plt.ylabel('MSE')
        #     plt.title('Training and Test MSE Convergence')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.show()
        
    def fit(self, train_data, test_data=None):
        if self.covariates is not None:
            self.x_train = torch.tensor(train_data[self.phi_columns + self.covariates].values, dtype=torch.float32)
            if test_data is not None:
                self.x_test = torch.tensor(test_data[self.phi_columns + self.covariates].values, dtype=torch.float32)
        else:
            self.x_train = torch.tensor(train_data[self.phi_columns].values, dtype=torch.float32)
            if test_data is not None:
                self.x_test = torch.tensor(test_data[self.phi_columns].values, dtype=torch.float32)

        self.y_train = torch.tensor(train_data['Density_gcm3'].values.reshape(-1, 1), dtype=torch.float32)
        if test_data is not None:
            self.y_test = torch.tensor(test_data['Density_gcm3'].values.reshape(-1, 1), dtype=torch.float32)

        self.model2 = DeepKriging(self.p)
        criterion = nn.MSELoss()
        if self.p > 10:
#           rate = 0.001
            rate = 0.001
        else:
#           rate = 0.001
            rate = 0.001
        optimizer = optim.Adam(self.model2.parameters(), lr=rate)
        self.train_losses = []
        if test_data is not None:
            self.train_model(self.model2, optimizer, criterion)
        else:
            for epoch in range(601):
                pre = self.model2(self.x_train)
                mse = criterion(pre, self.y_train)
                cost = mse
                
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                self.train_losses.append(mse.item())

    def predict(self, data):
        if self.covariates is not None:
            x = torch.tensor(data[self.phi_columns + self.covariates].values, dtype=torch.float32)
        else:
            x = torch.tensor(data[self.phi_columns].values, dtype=torch.float32)
        self.model2.eval()
        with torch.no_grad():
            predictions = self.model2(x).detach().numpy().flatten()
        return predictions

    def train_neural_network(self, mode="cross", test_size=None):
        if mode == "cross":
            # Define the number of folds for cross-validation
            num_folds = 10
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

            
            for fold, (train_index, test_index) in enumerate(kf.split(self.deposit_data)):
                self.train_data, self.test_data = self.deposit_data.iloc[train_index], self.deposit_data.iloc[test_index]

                if self.covariates is not None:
                    self.x_train = torch.tensor(self.train_data[self.phi_columns + self.covariates].values, dtype=torch.float32)
                    self.x_test = torch.tensor(self.test_data[self.phi_columns + self.covariates].values, dtype=torch.float32)
                else:
                    self.x_train = torch.tensor(self.train_data[self.phi_columns].values, dtype=torch.float32)
                    self.x_test = torch.tensor(self.test_data[self.phi_columns].values, dtype=torch.float32)

                self.y_train = torch.tensor(self.train_data['Density_gcm3'].values.reshape(-1, 1), dtype=torch.float32)
                self.y_test = torch.tensor(self.test_data['Density_gcm3'].values.reshape(-1, 1), dtype=torch.float32)

                self.model2 = DeepKriging(self.p)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(self.model2.parameters(), lr = 0.005)

                self.train_model(self.model2, optimizer, criterion)
                # Store metrics for this fold
                self.model2.eval()
                with torch.no_grad():
                    self.test_predictions_fold = self.model2(self.x_test).detach().numpy().flatten()
        
                self.test_mse_list.append(mean_squared_error(self.y_test, self.test_predictions_fold))
                self.test_mae_list.append(mean_absolute_error(self.y_test, self.test_predictions_fold))

                y_test_compat = self.test_data['Density_gcm3'].values

                n = len(y_test_compat)
                mean_y_test = np.mean(y_test_compat)
                sst = np.mean((y_test_compat - mean_y_test) ** 2)
                ssr = np.mean((self.test_predictions_fold - y_test_compat) ** 2)
                r2 = 1 - (ssr / sst)
                adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - self.p - 1))
                self.test_adjusted_r2_list.append(adjusted_r2)
                self.test_r2_list.append(r2)
            # Calculate and print average metrics across folds
            print("\nAverage Metrics Across Folds:")
            print(f"  Average MSE: {np.mean(self.test_mse_list):.4f}")
            print(f"  Average MAE: {np.mean(self.test_mae_list):.4f}")
            print(f"  Average Adjusted R2: {np.mean(self.test_adjusted_r2_list):.4f}")
            print(f"  Average R2: {np.mean(self.test_r2_list):.4f}")


        elif mode == "regular":
            if test_size is None:
                raise ValueError("For regular mode, test_size parameter must be specified.")

            self.train_data, self.test_data = train_test_split(self.deposit_data, test_size=test_size, random_state=42)

            if self.covariates is not None:
                self.x_train = torch.tensor(self.train_data[self.phi_columns + self.covariates].values, dtype=torch.float32)
                self.x_test = torch.tensor(self.test_data[self.phi_columns + self.covariates].values, dtype=torch.float32)
            else:
                self.x_train = torch.tensor(self.train_data[self.phi_columns].values, dtype=torch.float32)
                self.x_test = torch.tensor(self.test_data[self.phi_columns].values, dtype=torch.float32)

            self.y_train = torch.tensor(self.train_data['Density_gcm3'].values.reshape(-1, 1), dtype=torch.float32)
            self.y_test = torch.tensor(self.test_data['Density_gcm3'].values.reshape(-1, 1), dtype=torch.float32)

            self.x_train_df = pd.DataFrame(self.x_train.numpy(), columns=self.phi_columns + self.covariates if self.covariates else self.phi_columns)
            self.x_test_df = pd.DataFrame(self.x_test.numpy(), columns=self.phi_columns + self.covariates if self.covariates else self.phi_columns)

            self.model2 = DeepKriging(self.p)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model2.parameters(), lr = 0.005)

            self.train_model(self.model2, optimizer, criterion)

            self.model2.eval()
            with torch.no_grad():
                self.test_predictions = self.model2(self.x_test).detach().numpy().flatten()
    
            self.print_metrics(self.y_test, self.test_predictions, "Deposit Data")

    def print_metrics(self, true_values, predicted_values, dataset_name):
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)
        print(f"{dataset_name} MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


