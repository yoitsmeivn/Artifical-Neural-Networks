import torch
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer


class Regressor():

    def shuffle_split(dataset):

        #important rows should remain together when shuffling the dataset

        shuffled = dataset.sample(frac=1, random_state=50).reset_index(drop=True)
        group_size = int(len(dataset) * 0.8)

        input = shuffled.iloc[:group_size] 
        target = shuffled[group_size:] 

        input_set_x = input.iloc[:, :-1]
        input_set_y = input.iloc[:, -1]
        target_set_x = target.iloc[:, :-1]
        target_set_y = target.iloc[:, -1]
    
        return input_set_x, input_set_y, target_set_x, target_set_y

    def __init__(self, x, nb_epoch = 10, learning_rate = 0.0001, batch_size = 32, hidden_values= [16,32]):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own

        self.lbinarizer = {}
        self.data = {}

        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 

        #own
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_values = hidden_values
    
    

        self.neural_network = torch.nn.Sequential( 
            torch.nn.Linear(self.input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(4, self.output_size),
        )

        self.loss = torch.nn.MSELoss()

        self.optim = torch.optim.Adam(self.neural_network.parameters(), lr = self.learning_rate)
        
        
        
        
        

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        #filling in empty gaps with 0--according to spec
        x = x.copy().fillna(0)

        #encoding the categorical variables (cessentially converinting it to numbers)
        categories = x.select_dtypes(include=['object']).columns
        numbers = x.select_dtypes(include=['number']).columns

        for each in categories:
            if training:
                lb = LabelBinarizer()
                change_variable = lb.fit_transform(x[each])
                self.lbinarizer[each] = lb
            else:
                lb = self.lbinarizer[each]
                change_variable = lb.transform(x[each]) if lb else np.zeros((len(x), 1))

            x = pd.concat([x.drop(columns=[each]), pd.DataFrame(change_variable, index=x.index)], axis=1)
        
        
        # z normalization = (x - mean) / stdev
        
        for each in numbers:
            if training:
                mu = x[each].mean()
                sigma = x[each].std()
                self.data[each] = (mu, sigma)
            else:
                mu, sigma = self.data.get(each, (0,1))
            
            #edge case - sigma cannot be 0 
            if sigma == 0:
                x[each] = (x[each] - mu)
            else:
                x[each] = (x[each] - mu) / sigma
        
        
        if y is not None:
            y = y.astype(float)
            
        x_var_tensor = torch.tensor(x.select_dtypes(include=np.number).values, dtype=torch.float32)
        
        y_var_tensor = torch.tensor(y.values / 100000, dtype=torch.float32)  if y is not None else None 

        return x_var_tensor, y_var_tensor
        # Return preprocessed x and y, return None for y if it was None
        # return x, (y if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y, val_x=None, val_y=None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        if val_x is not None and val_y is not None:
            validation_X, validation_Y = self._preprocessor(val_x, y=val_y, training=False)

        #end training if the loss value stops to improve
        smallest_val_loss = float("inf")
        epoch_limit = 8
        epoch_iterations = 0

        #epochs are like the number of times a model is trained on a dataset, so use a for loop to iterate through every value within the range of epochs
        #helper function to stop training early if no improvements are being made
        def early_stop(loss_validation, smallest_val_loss, epoch_iterations, learning_rate):
            if loss_validation > smallest_val_loss:
                epoch_iterations += 1
                #reducing the learning rate for every epoch to  finetine training near the smallest loss value according to module 5 lecture
                learning_rate *= 0.75
            else:
                smallest_val_loss = loss_validation
                epoch_iterations = 0

            return epoch_iterations >= epoch_limit
                
        
        #for every iteration of training or in other words every epoch
        for each in range(self.nb_epoch):
            self.neural_network.train()
            for first in range(0, X.size(0), self.batch_size):
                last = first + self.batch_size
                x_batch, y_batch = X[first:last], Y[first:last].view(-1, 1)

                self.optim.zero_grad()
                #forward pass
                estimations = self.neural_network(x_batch)
                #calculates the loss after the forward pass
                loss_values = self.loss(estimations, y_batch)
                #backwards pass
                loss_values.backward()
                #gradient descent
                self.optim.step()
            if val_x is not None and val_y is not None:
                self.neural_network.eval()
                with torch.no_grad():
                    estimations_validation = self.neural_network(validation_X)
                    loss_validation = self.loss(estimations_validation, validation_Y).item()


                if early_stop(loss_validation=loss_validation, smallest_val_loss=smallest_val_loss, epoch_iterations=epoch_iterations, learning_rate=self.learning_rate):
                    break

    
            
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        self.neural_network.eval()

        #dont need any gradient computation for determining predictions
        with torch.no_grad():
            estimations = self.neural_network(X)

        return estimations.numpy() * 100000 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y=y, training = False) # Do not forget


        with torch.no_grad():
            estimations = self.neural_network(X)
            #average of (diff between prediction and actual)^2
            
            mean_sq_error = torch.mean((Y - estimations) ** 2).item()

        #R^2 calculations
            residual_sum = torch.sum((Y - estimations)**2)
            total = torch.sum((Y - torch.mean(Y))**2).item()
            ratio = residual_sum/total

            r_score = 1 - ratio

        

        return mean_sq_error*100000

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def perform_hyperparameter_search(training_group_x, training_group_y, val_group_x, val_group_y): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    optimal_hypeparam = None
    optimal_score = float('inf')



    learn_rates = [ 0.001, 0.01, 0.1]
    hidden_values = [4, 8, 16, 32, 64, 128]
    epochs = [10, 50, 100, 150, 200]
    batches = [32, 64]

    #l
    for rate in learn_rates:
        for values in hidden_values:
            for batch in batches:
                for epoch in epochs:
                    regresssor_model = Regressor(training_group_x, nb_epoch=epoch, learning_rate=rate, batch_size=batch, hidden_values=values)
                    # fit_model = regresssor_model.fit(training_group_x, training_group_y)
                    
                    # predictoin_model = fit_model.predict(val_set_x)
                    score_model_mse = regresssor_model.score(val_group_x, val_group_y, check=False)
            
    
                    
                    if optimal_score > score_model_mse:
                        optimal_score = score_model_mse
                        optimal_hypeparam = {rate, values, epoch, batch}
                        
    #print(f" The best hyperparameters (learning rate, hidden values, epochs, batch size) are: {optimal_hypeparam} with an MSE of {optimal_score}")
    return  optimal_hypeparam # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():



    output_label = "median_house_value"



    # Use pandas to read CSV data as it contains various object types

    # Feel free to use another CSV reader tool

    # But remember that LabTS tests take Pandas DataFrame as inputs

    data = pd.read_csv("housing.csv") 



    # Splitting input and output

    x_train = data.loc[:, data.columns != output_label]

    y_train = data.loc[:, [output_label]]



    # Training

    # This example trains on the whole available dataset. 

    # You probably want to separate some held-out data 

    # to make sure the model isn't overfitting

    regressor = Regressor(x_train, nb_epoch = 10)

    regressor.fit(x_train, y_train)

    save_regressor(regressor)



    # Error

    error = regressor.score(x_train, y_train)

    print(f"\nRegressor error: {error}\n")


if __name__ == "__main__":
    example_main()