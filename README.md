Email Spam Classification using Naive Bayes

This project performs email spam classification using the Naive Bayes algorithm. It uses a dataset of email messages labeled as spam or ham (non-spam) to train a Naive Bayes model. The model is then used to classify new email messages as either spam or ham.

### Steps involved:
1. Importing the necessary libraries
2. Loading and preprocessing the dataset
3. Text preprocessing
4. Creating a bag-of-words representation
5. Splitting the data into training and testing sets
6. Training the Naive Bayes model
7. Evaluating the model
8. Custom message classification

### Model used:
The Naive Bayes algorithm is employed for email spam classification. Specifically, the Multinomial Naive Bayes variant is used, which is suitable for discrete features like word counts. It calculates the likelihood of a given email message belonging to a particular class (spam or ham) based on the probabilities of words occurring in spam and ham messages.

### Usage:
To use the code, you need to have the required libraries installed. The email spam dataset should be provided in a CSV file format, where each row contains an email message and its corresponding label. The code preprocesses the dataset, trains the model, and provides functionality to classify new email messages as spam or ham.

Feel free to modify and experiment with the code to suit your specific requirements.

