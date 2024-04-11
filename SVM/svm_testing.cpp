#include <iostream>
#include <vector>
#include <random>
#include "SVM.h"

using namespace std;

int main()
{
    // Generating random data
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-10, 10);

    vector<vector<double>> X_data(20, vector<double>(2));
    vector<vector<double>> y_data(20, vector<double>(1));

    for (int i = 0; i < 20; ++i)
    {
        X_data[i][0] = dis(gen);                                   // Feature 1
        X_data[i][1] = dis(gen);                                   // Feature 2
        y_data[i][0] = (X_data[i][0] + X_data[i][1] > 0) ? 1 : -1; // Binary labels
    }

    // Creating Matrix objects
    Matrix X(X_data);
    Matrix y(y_data);

    // SVM Classifier
    SVM svm;       // Assuming default constructor is appropriate
    svm.fit(X, y); // Train the SVM with the data

    // Predicting using the trained model
    vector<vector<double>> new_point_data = {{dis(gen), dis(gen)}};
    Matrix new_point_matrix(new_point_data);

    Matrix prediction = svm.predict(new_point_matrix);
    cout << "Predicted class for the new point: " << prediction.return_row(0)[0] << endl; // Adjusted to handle Matrix return type

    // Displaying the matrix data
    cout << "Feature Matrix X:" << endl;
    X.printMatrix();
    cout << "\nLabel Matrix y:" << endl;
    y.printMatrix();
    cout << "\nNew Data Point:" << endl;
    new_point_matrix.printMatrix();
    cout << endl;

    return 0;
}
