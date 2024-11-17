import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for non-GUI environments
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                  [cluster_std * 0.8, cluster_std]])
    
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    
    X2 += [distance, -distance]
    y2 = np.ones(n_samples)

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y


def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def do_experiments(start, end, step_num):
    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []

    sample_data = {}
    n_samples = 8

    n_cols = 2  # Fixed number of columns

    n_rows = (n_samples + n_cols - 1) // n_cols  # Calculate rows needed

    plt.figure(figsize=(20, n_rows * 10))  # Adjust figure height based on rows

    # Run experiments for each shift distance

    for i, distance in enumerate(shift_distances, 1):

        X, y = generate_ellipsoid_clusters(distance=distance)

        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)

        

        # Record parameters
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)



        # Calculate and store slope and intercept

        slope = -beta1 / beta2

        intercept = -beta0 / beta2

        slope_list.append(slope)

        intercept_list.append(intercept)



        # Plot dataset

        plt.subplot(n_rows, n_cols, i)

        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0')

        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1')



        # Calculate margin width between 70% confidence contours for each class

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        Z = Z.reshape(xx.shape)

        

        # Plot confidence contours

        contour_levels = [0.7, 0.8, 0.9]

        alphas = [0.05, 0.1, 0.15]

        for level, alpha in zip(contour_levels, alphas):

            class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)

            class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)

            if level == 0.7:

                distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices,

                                class_0_contour.collections[0].get_paths()[0].vertices,

                                metric='euclidean')

                min_distance = np.min(distances)

                margin_widths.append(min_distance)



        plt.title(f"Shift Distance = {distance:.2f}", fontsize=24)

        plt.xlabel("x1")

        plt.ylabel("x2")

        

        # Add equation and margin width text

        equation_text = f"{beta0:.2f} + {beta1:.2f} * x1 + {beta2:.2f} * x2 = 0\nx2 = {slope:.2f} * x1 + {intercept:.2f}"

        margin_text = f"Margin Width: {min_distance:.2f}"

        plt.text(x_min + 0.1, y_max - 1.0, equation_text, fontsize=24, color="black",

                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        plt.text(x_min + 0.1, y_max - 1.5, margin_text, fontsize=24, color="black",

                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))



        if i == 1:

            plt.legend(loc='lower right', fontsize=20)



        sample_data[distance] = (X, y, model, beta0, beta1, beta2, min_distance)



    plt.tight_layout()

    plt.savefig(f"{result_dir}/dataset.png")



    # Plot parameters vs. shift distance

    plt.figure(figsize=(18, 15))

    

    # Beta0 plot

    plt.subplot(3, 3, 1)

    plt.plot(shift_distances, beta0_list, 'b-')

    plt.title("Shift Distance vs Beta0")

    plt.xlabel("Shift Distance")

    plt.ylabel("Beta0")

    

    # Beta1 plot

    plt.subplot(3, 3, 2)

    plt.plot(shift_distances, beta1_list, 'r-')

    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")

    plt.xlabel("Shift Distance")

    plt.ylabel("Beta1")

    

    # Beta2 plot

    plt.subplot(3, 3, 3)

    plt.plot(shift_distances, beta2_list, 'g-')

    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")

    plt.xlabel("Shift Distance")

    plt.ylabel("Beta2")

    

    # Slope plot

    plt.subplot(3, 3, 4)

    plt.plot(shift_distances, slope_list, 'm-')

    plt.title("Shift Distance vs Beta1 / Beta2 (Slope)")

    plt.xlabel("Shift Distance")

    plt.ylabel("Beta1 / Beta2")

    plt.ylim(-2, 0)

    

    # Intercept ratio plot

    plt.subplot(3, 3, 5)

    plt.plot(shift_distances, [-b0/b2 for b0, b2 in zip(beta0_list, beta2_list)], 'c-')

    plt.title("Shift Distance vs Beta0 / Beta2 (Intercept Ratio)")

    plt.xlabel("Shift Distance")

    plt.ylabel("Beta0 / Beta2")

    

    # Margin width plot

    plt.subplot(3, 3, 7)

    plt.plot(shift_distances, margin_widths, 'k-')

    plt.title("Shift Distance vs Margin Width")

    plt.xlabel("Shift Distance")

    plt.ylabel("Margin Width")

    

    plt.tight_layout()

    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")



if __name__ == "__main__":

    start = 0.25

    end = 2.0

    step_num = 8

    do_experiments(start, end, step_num)

















