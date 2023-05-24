import pickle
import time

from math import sqrt

import numpy as np
from scipy.special import logsumexp

from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
from functools import partial

import optuna

class GreedyKernelLearner(object):
    """Greedy Kernel learner class

    Parameters
    ----------
    dataset: dict
        The dataset as a dictionnary with the following keys:
        X_train, X_valid, X_test, y_train, y_valid, y_test, name.

    C_range: list
        C values range to search from (SVM's penalty parameter).
        Used while learning a linear classifier over the mapped dataset.

    gamma: float
        Gamma value (RBF kernel's bandwith parameter).
        Used for sampling the Fourier features.

    N: int
        Number of points to sample from the Fourier transform distribution.

    random_state: None, int or instance of RandomState.
        Information about the random state to be used.


    Attributes
    ----------
    dataset : dict
        The dataset as a dictionnary with the following keys:
        X_train, X_valid, X_test, y_train, y_valid, y_test, name.

    n : int
        Number of samples in the training set (X_train.shape[0]).

    d : int
        Number of features in the dataset (X_train.shape[1]).

    C_range : list
        C values range to search from (SVM's penalty parameter).

    gamma : float
        Gamma value (RBF kernel's bandwith parameter).

    sigma: float
        Sigma value computed using gamma value: sigma = 1 / sqrt(2 * gamma)

    N: int
        Number of points to sample from the Fourier transform distribution.

    random_state: instance of RandomState.
        Random state for all random operations.

    loss: array, shape = [N,]
        Empirical losses matrix.

    time: list
        List of all computation steps times as a tuple (step name, time (s))

    omega : array, shape = [d, N]
        omega vector sampled from the Fourier distribution.

    beta : float
        Beta value (pseudo-posterior "temperature" parameter).

    pb_Q : array, shape = [N,]
        PAC-Bayesian Pseudo-posterior distributions over the features.

    rho : float
        Optimized kernel method parameter, act as a constraint in the optimization problem.

    ok_Q : array, shape = [N,]
        Optimized Kernel distributions over the features.
    """
    def __init__(self, dataset, C_range, maxTry_range, p_range, epsilon_range, gamma, N, random_state=42):
        self.dataset = dataset
        self.n, self.d = self.dataset['X_train'].shape
        self.C_range = C_range
        self.gamma = gamma
        self.sigma = 1. / sqrt(2 * self.gamma)
        self.N = N
        self.random_state = check_random_state(random_state)
        self.loss = None
        self.time = []
        self.p_range = p_range # Ajout de la liste de l'hyperparamètre p par Jacob
        self.maxTry_range = maxTry_range # Ajout de la liste de l'hyperparamètre maxTry par Jacob
        self.epsilon_range = epsilon_range # Ajout de la liste de l'hyperparamètre epsilon par Jacob

    def sample_omega(self):
        """Randomly sample omega."""
        start_time = time.time()
        self.omega = self.random_state.randn(self.d, self.N) / self.sigma
        self.time.append(("sampling", (time.time() - start_time) * 1000))

    def compute_loss(self):
        """Compute empirical losses matrix."""
        start_time = time.time()
        cos_values = np.sum(np.einsum('ij,i->ij', self.transform_cos(self.omega, self.dataset['X_train']), self.dataset['y_train']), axis=0)
        sin_values = np.sum(np.einsum('ij,i->ij', self.transform_sin(self.omega, self.dataset['X_train']), self.dataset['y_train']), axis=0)

        self.loss = 1 / (self.n * (self.n - 1)) * (cos_values ** 2 + sin_values ** 2)
        self.loss = (1 - self.loss) / 2 + 1 / (2 * (self.n -1))
        self.time.append(("loss", (time.time() - start_time) * 1000))

    def learn_rff(self, D):
        """Learn using classical Random Fourier Features method.

        Parameters
        ----------
        D: int
            Number of Fourier features to use.

        Returns
        -------
        results: dict
            Relevant metrics and informations.
        """
        kernel_features = self.omega[:, :D]

        transformed_X_train = self.transform_sincos(kernel_features, self.dataset['X_train'], D)
        transformed_X_valid = self.transform_sincos(kernel_features, self.dataset['X_valid'], D)
        transformed_X_test = self.transform_sincos(kernel_features, self.dataset['X_test'], D)

        # C search using a validation set
        C_search = []
        start_time = time.time()
        for C in self.C_range:
            clf = LinearSVC(C=C, random_state=self.random_state)
            clf.fit(transformed_X_train, self.dataset['y_train'])
            err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))
            C_search.append((err, C, clf))
        self.time.append(("learning", (time.time() - start_time) * 1000))

        # Computing relevant metrics
        val_err, C, clf = sorted(C_search, key=lambda x: x[0])[0]
        train_err = 1 - accuracy_score(self.dataset['y_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_test'], y_pred)
        f1 = f1_score(self.dataset['y_test'], y_pred)

        return dict([("dataset", self.dataset['name']), ("exp", 'greedy'), ("algo", 'RFF'), ("C", C), ("D", D), ("N", self.N), \
                    ("gamma", self.gamma), ("train_error", train_err), ("val_error", val_err), ("test_error", test_err), ("f1", f1), \
                    ("time", self.time)])


    def compute_pb_Q(self, beta):
        """Compute PAC-Bayesian pseudo-posterior Q distribution over the Fourier features.

        Parameters
        ----------
        beta: float
            Beta value (pseudo-posterior "temperature" parameter).
        """
        start_time = time.time()
        t = sqrt(self.n) * beta

        self.beta = beta
        self.pb_Q = -t*self.loss - logsumexp(-t*self.loss)
        self.pb_Q = np.exp(self.pb_Q)
        self.time.append(("pb_Q", (time.time() - start_time) * 1000))

    # Ajout de l'algorithme vorace greedy
    # ----------------------------------------------------------------------------------------

    def greedy_pbrff(self, Dmax, maxTry, p, epsilon):
        """
        Greedy algorithm that use PAC-Bayes techniques to choose the kernel_features.

        Parameters
        ----------
        omega : array, shape = [d, N]
            omega vector sampled from the Fourier distribution.

        pb_Q : array, shape = [N,]
            PAC-Bayesian pseudo-posterior Q distribution over the Fourier features.

        Dmax : int
            Maximum number of RFF.

        maxTry : int
            Maximum number of tries.

        p : int
            Number of RFF choose at each step p.

        loss : array, shape = [N,]
            Empirical losses matrix.

        epsilon : int
            Hyperparameter.

        Returns
        -------
        H : array
            Array that contains index of RFF in the array Omega.
        """

        loss_list = []
        E_list = []

        # test = (p / (2 * self.N)) - (np.sum(self.loss[self.random_state.choice(self.omega.shape[1], p, replace=True, p=self.pb_Q)]) * (1 / self.N))
        # print(f"Sum loss : {test}")
        # print(f"Step : {10**(np.round(np.log10(test)))}")

        H = np.array([]) # Initialize as empty array 
        Try = 0 

        while H.size < Dmax:
            
            if H.size + p <= Dmax:
                # Pick p RFF Ws = {s = 1, ..., p}
                rff_index = self.random_state.choice(self.omega.shape[1], p, replace=True, p=self.pb_Q)
            else:
                # Pick Dmax - H.size RFF
                rff_index = self.random_state.choice(self.omega.shape[1], Dmax - H.size, replace=True, p=self.pb_Q)

            # Calculate E 
            loss_value = self.loss[rff_index]
            loss_list.append(loss_value)
            E = (p / (2 * self.N)) - ((1 / self.N) * np.sum(loss_value))
            E_list.append(E)

            if E >= epsilon:
                H = np.concatenate((H, rff_index), axis=0)
                #H = H + rff_index.tolist()
                Try = 0
            else:
                if Try >= maxTry:
                    break
                else:
                    Try += 1

        return H.size, H.astype(int), loss_list, E_list

    # ----------------------------------------------------------------------------------------

    # Ajout de la méthode pour faire l'optimisation bayésienne pour choisir les hyperparamètres C, p, maxTry et epsilon
    # ----------------------------------------------------------------------------------------

    def bayesian_optimization_(self, trial, D, method):

        if method == "base":
            kernel_features = self.omega[:, self.random_state.choice(self.omega.shape[1], D, replace=True, p=self.pb_Q)]
        
        elif method == "greedy":
            maxTry = trial.suggest_int("maxTry", 2, 10)
            p = trial.suggest_int("p", 1, 50)
            epsilon = trial.suggest_float("epsilon", 1e-5 , 3e-1, log=True)
            # E_avg = (p / (2 * self.N)) - (np.sum(self.loss[self.random_state.choice(self.omega.shape[1], p, replace=True, p=self.pb_Q)]) * (1 / self.N))
            # eps_min = E_avg - (10**(np.round(np.log10(E_avg))))
            # eps_max = E_avg + (10**(np.round(np.log10(E_avg))))
            # epsilon = trial.suggest_float("epsilon", eps_min, eps_max, step=(10**(np.round(np.log10(E_avg)))))
            # epsilon = E_avg - (10**(np.round(np.log10(E_avg))))
            # print(f"Epsilon : {epsilon}")
            _, index, _, _ = self.greedy_pbrff(D, maxTry, p, epsilon)
            if index.size == 0:
                return 1
            else:
                kernel_features = self.omega[:, index]
            #kernel_features = self.omega[:, index]

        
        transformed_X_train = self.transform_sincos(kernel_features, self.dataset['X_train'], D)
        transformed_X_valid = self.transform_sincos(kernel_features, self.dataset['X_valid'], D)
        clf = LinearSVC(C=trial.suggest_float("C", 1, 10000), random_state=self.random_state)
        #clf = LinearSVC(C=100, random_state=self.random_state)
        clf.fit(transformed_X_train, self.dataset['y_train'])
        return 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))

    
    def bayesian_optimiation(self, D, method):
        
        study = optuna.create_study()
        f = partial(self.bayesian_optimization_, method=method, D=D)
        study.optimize(f, n_trials=100)

        if method == "base":
            C = study.best_params['C']
            return C
        
        elif method == "greedy":
            maxTry = study.best_params['maxTry']
            p = study.best_params['p']
            epsilon = study.best_params['epsilon']
            # E_avg = (p / (2 * self.N)) - (np.sum(self.loss[self.random_state.choice(self.omega.shape[1], p, replace=True, p=self.pb_Q)]) * (1 / self.N))
            # epsilon = E_avg - (10**(np.round(np.log10(E_avg))))
            C = study.best_params['C']
            #C = 100

            return [C, maxTry, p, epsilon]


    # ----------------------------------------------------------------------------------------

    # Ajout de la méthode pour le GridSearch pour choisir les hyperparamètres C, p, maxTry et epsilon
    # ----------------------------------------------------------------------------------------

    def gridSearch(self, D, method):

        if method == "greedy":
            param_list = []

            start_time = time.time()
            for maxTry in self.maxTry_range:
                for p in self.p_range:
                    for epsilon in self.epsilon_range:
                        for C in self.C_range:
            
                            kernel_features = self.omega[:, self.greedy_pbrff(D, maxTry, p, epsilon)[1]]

                            transformed_X_train = self.transform_sincos(kernel_features, self.dataset['X_train'], D)
                            transformed_X_valid = self.transform_sincos(kernel_features, self.dataset['X_valid'], D)

                            clf = LinearSVC(C=C, random_state=self.random_state)
                            clf.fit(transformed_X_train, self.dataset['y_train'])
                            err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))
                            param_list.append((err, [C, maxTry, p, epsilon], clf))

            self.time.append(("learning", (time.time() - start_time) * 1000))

            _, best_params, clf = sorted(param_list, key=lambda x: x[0])[0]

            return best_params
        
        elif method == "base":

            kernel_features = self.omega[:, self.random_state.choice(self.omega.shape[1], D, replace=True, p=self.pb_Q)]

            transformed_X_train = self.transform_sincos(kernel_features, self.dataset['X_train'], D)
            transformed_X_valid = self.transform_sincos(kernel_features, self.dataset['X_valid'], D)

            # C search using a validation set
            C_search = []
            start_time = time.time()
            for C in self.C_range:
                clf = LinearSVC(C=C, random_state=self.random_state)
                clf.fit(transformed_X_train, self.dataset['y_train'])
                err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))
                C_search.append((err, C, clf))
            self.time.append(("learning", (time.time() - start_time) * 1000))

            # Computing relevant metrics
            _, C, clf = sorted(C_search, key=lambda x: x[0])[0]
        
            return C


    # ----------------------------------------------------------------------------------------

    def learn_pbrff(self, D, method, C, maxTry=0, p=0, epsilon=0):
        """Learn using PAC-Bayes Random Fourier Features method

        Parameters
        ----------
        D: int
            Number of Fourier features to subsample.

        method: String
            Method to choose between base or greedy.

        Returns
        -------
        results: dict
            Relevant metrics and informations.
        """

        if method == "base":
            # Pour le modèle de base
            kernel_features = self.omega[:, self.random_state.choice(self.omega.shape[1], D, replace=True, p=self.pb_Q)]
        elif method == "greedy":
            # Pour le modèle avec l'algorithme vorace
            nbrD_choisi, index, loss_list, E_list = self.greedy_pbrff(D, maxTry, p, epsilon)
            print(f"Nbr d'index : {index}")
            if index.size == 0:
                kernel_features = self.omega[:, self.random_state.choice(self.omega.shape[1], D, replace=True, p=self.pb_Q)]
            else:
                kernel_features = self.omega[:, index]

        transformed_X_train = self.transform_sincos(kernel_features, self.dataset['X_train'], D)
        transformed_X_valid = self.transform_sincos(kernel_features, self.dataset['X_valid'], D)
        transformed_X_test = self.transform_sincos(kernel_features, self.dataset['X_test'], D)

        # # C search using a validation set
        # C_search = []
        # start_time = time.time()
        # for C in self.C_range:
        #     clf = LinearSVC(C=C, random_state=self.random_state)
        #     clf.fit(transformed_X_train, self.dataset['y_train'])
        #     err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))
        #     C_search.append((err, C, clf))
        # self.time.append(("learning", (time.time() - start_time) * 1000))


        clf = LinearSVC(C=C, random_state=self.random_state)
        clf.fit(transformed_X_train, self.dataset['y_train'])
        err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))

        # Computing relevant metrics
        val_err = err
        train_err = 1 - accuracy_score(self.dataset['y_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_test'], y_pred)
        f1 = f1_score(self.dataset['y_test'], y_pred)

        if method == "base":
            # Pour le modèle de base
            return dict([("dataset", self.dataset['name']), ("exp", 'greedy'), ("algo", 'PBRFF'), ("C", C), ("D", D), ("N", self.N), \
                    ("gamma", self.gamma), ("beta", self.beta), ("train_error", train_err), ("val_error", val_err), \
                    ("test_error", test_err), ("f1", f1), ("time", self.time)])
        elif method == "greedy":
            # Pour le modèle avec l'algorithme vorace
            return dict([("dataset", self.dataset['name']), ("exp", 'greedy'), ("algo", 'PBRFF'), ("C", C), ("D", D), ("N", self.N), \
                    ("gamma", self.gamma), ("beta", self.beta), ("Loss_l", loss_list), ("E_l", E_list), ("nbrD_choisi", nbrD_choisi), ("maxTry", maxTry), ("p", p), ("epsilon", epsilon), ("train_error", train_err), ("val_error", val_err), \
                    ("test_error", test_err), ("f1", f1), ("time", self.time)])


    def compute_ok_Q(self, rho):
        """Compute Optimized Kernel distribution over the Fourier features as implemented by Sinha and Duchi 2016
        in matlab (https://github.com/amansinha/learning-kernels, linear_chi_square function)

        Parameters
        ----------
        rho: float
            Optimized kernel method parameter, act as a constraint in the optimization problem.
        """

        start_time = time.time()
        self.rho = rho
        v = 2 * self.loss -1
        u = np.ones(self.N) * 1 / self.N
        acc = 1e-8

        def project_onto_simplex(w, B):
            z = -np.sort(-w)
            sv = np.cumsum(z)
            rho = np.argwhere(z > np.divide((sv - B), np.arange(1, len(w) + 1)))[-1][0]
            theta = (sv[rho] - B) / (rho + 1)
            q = w - theta
            q[q < 0] = 0

            return q

        duality_gap = np.inf

        max_lambda = np.inf
        min_lambda = 0

        x = project_onto_simplex(u, 1)

        if (np.linalg.norm(u-x) ** 2 > rho):
            raise RuntimeError("Problem is not feasible")

        start_lambda = 1
        while(np.isinf(max_lambda)):
            x = project_onto_simplex(u - v / start_lambda, 1)
            lam_grad = 0.5 * np.linalg.norm(x - u) ** 2 - self.rho/2;
            if (lam_grad < 0):
                max_lambda = start_lambda;
            else:
                start_lambda = start_lambda * 2;

        while (max_lambda - min_lambda > acc * start_lambda):
            lambda_value = (min_lambda + max_lambda) / 2
            x = project_onto_simplex(u - v / lambda_value, 1);
            lam_grad = 0.5 * np.linalg.norm(x - u) ** 2 - self.rho/2;
            if (lam_grad < 0):
            # Then lambda is too large, so decrease max_lambda
                max_lambda = lambda_value
            else:
                min_lambda = lambda_value

        self.ok_Q = x
        self.time.append(("ok_Q", (time.time() - start_time) * 1000))

    def learn_okrff(self, D):
        """Learn using Optimized Kernel Random Fourier Features method from Sinha et Duchi (2016)

        Parameters
        ----------
        D: int
            Number of Fourier features to subsample.

        Returns
        -------
        results: dict
            Relevant metrics and informations.
        """
        kernel_features = self.omega[:, self.random_state.choice(self.omega.shape[1], D, replace=True, p=self.ok_Q)]

        transformed_X_train = self.transform_sincos(kernel_features, self.dataset['X_train'], D)
        transformed_X_valid = self.transform_sincos(kernel_features, self.dataset['X_valid'], D)
        transformed_X_test = self.transform_sincos(kernel_features, self.dataset['X_test'], D)

        # C search using a validation set
        C_search = []
        start_time = time.time()
        for C in self.C_range:
            clf = LinearSVC(C=C, random_state=self.random_state)
            clf.fit(transformed_X_train, self.dataset['y_train'])
            err = 1 - accuracy_score(self.dataset['y_valid'], clf.predict(transformed_X_valid))
            C_search.append((err, C, clf))
        self.time.append(("learning", (time.time() - start_time) * 1000))

        # Computing relevant metrics
        val_err, C, clf = sorted(C_search, key=lambda x: x[0])[0]
        train_err = 1 - accuracy_score(self.dataset['y_train'], clf.predict(transformed_X_train))
        y_pred = clf.predict(transformed_X_test)
        test_err = 1 - accuracy_score(self.dataset['y_test'], y_pred)
        f1 = f1_score(self.dataset['y_test'], y_pred)

        return dict([("dataset", self.dataset['name']), ("exp", 'greedy'), ("algo", 'OKRFF'), ("C", C), ("D", D), ("N", self.N), \
                    ("gamma", self.gamma), ("rho", self.rho), ("train_error", train_err), ("val_error", val_err), \
                    ("test_error", test_err), ("f1", f1), ("time", self.time)])

    def transform_sincos(self, omega, X, D):
        """Example mapping: phi(x)

        Parameters
        ----------
        omega : array, shape = [d, D]
            omega vector sampled from the Fourier distribution.

        X: array, shape = [n, d]
            Samples.

        D: int
            Number of Fourier features subsampled.

        Returns
        -------
        mapped_X: array, shape = [n, 2D]
            Mapped samples X.
        """
        wX = np.dot(X, omega)
        return np.hstack((np.cos(wX), np.sin(wX))) / np.sqrt(D)

    def transform_cos(self, omega, delta):
        """Hypothesis computation with cos: h_omega(delta)

        Parameters
        ----------
        omega: array, shape = [d, D]
            omega values (sampled from the Fourier features).

        delta: array, shape = [n, d]
            Pairwise distances.

        Returns
        -------
        hypothesis: array, shape = [n, D]
            Hypothesis values.
        """
        return np.cos(np.dot(delta, omega))

    def transform_sin(self, omega, delta):
        """Hypothesis computation with sin: h_omega(delta)

        Parameters
        ----------
        omega: array, shape = [d, D]
            omega values (sampled from the Fourier features).

        delta: array, shape = [n, d]
            Pairwise distances.

        Returns
        -------
        hypothesis: array, shape = [n, D]
            Hypothesis values.
        """
        return np.sin(np.dot(delta, omega))

def compute_greedy_kernel(args, greedy_kernel_learner_file, gamma, D_range, random_state):
    """Greedy kernel learning function for parallel processing."""
    tmp_results = []

    with open(greedy_kernel_learner_file, 'rb') as in_file:
        greedy_kernel_learner = pickle.load(in_file)

    if args["algo"] == "rff":
        print("Processing: rff")
        for D in D_range:
            tmp_results.append(greedy_kernel_learner.learn_rff(D))

    elif args["algo"] == "pbrff":
        print(f"Processing: pbrff with beta: {args['param']}")
        greedy_kernel_learner.compute_pb_Q(beta=args['param'])
        
        for D in D_range:
            # GridSearch
            #best_params = greedy_kernel_learner.gridSearch(D, "greedy")
            
            # C'est ici qu'on va appeler l'optimisation bayésienne
            best_params = greedy_kernel_learner.bayesian_optimiation(D, "greedy")

            #tmp_results.append(greedy_kernel_learner.learn_pbrff(D, "base", 100))
            tmp_results.append(greedy_kernel_learner.learn_pbrff(D, "greedy", best_params[0], best_params[1], best_params[2], best_params[3]))

    elif args["algo"] == "okrff":
        print(f"Processing: okrff with rho: {args['param']}")
        greedy_kernel_learner.compute_ok_Q(rho=args['param'])
        for D in D_range:
            tmp_results.append(greedy_kernel_learner.learn_okrff(D))


    with open(args["output_file"], 'wb') as out_file:
        pickle.dump(tmp_results, out_file, protocol=4)

    return args["algo"]
