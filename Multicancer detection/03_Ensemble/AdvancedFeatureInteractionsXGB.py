import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from scipy import stats
import logging
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


class AdvancedFeatureInteractions:
    """Feature Interaction Processing Class"""

    def __init__(self,
                 interaction_method='gaussian',
                 selection_method='mutual_info',
                 max_features=30,
                 gaussian_gamma='auto',
                 scale_after_interaction=True):
        """
        Initialize the feature interaction processor

        Parameters:
            interaction_method: Feature interaction method ('gaussian', 'polynomial', 'custom', 'combined')
            selection_method: Feature selection method ('mutual_info', 'correlation', 'pca', None)
            max_features: Maximum number of features to select
            gaussian_gamma: Gamma parameter for Gaussian kernel, can be 'auto' or a specific value
            scale_after_interaction: Whether to standardize after interaction
        """
        self.interaction_method = interaction_method
        self.selection_method = selection_method
        self.max_features = max_features
        self.gaussian_gamma = gaussian_gamma
        self.scale_after_interaction = scale_after_interaction

        # Initialize other attributes
        self.feature_names_ = None
        self.selected_features_ = 30
        self.feature_importance_ = None
        self.scaler = StandardScaler()

        # Save post-interaction scaler and feature selector
        self.interaction_scaler = None
        self.selector = None

        # Record if it has been fitted
        self.is_fitted = False

        # Output initialization parameters
        logging.info("Initialized AdvancedFeatureInteractions with:")
        logging.info("- interaction_method: {}".format(self.interaction_method))
        logging.info("- selection_method: {}".format(self.selection_method))
        logging.info("- max_features: {}".format(self.max_features))
        logging.info("- gaussian_gamma: {}".format(self.gaussian_gamma))

    def set_original_feature_names(self, names):
        """Set the names of original features"""
        self.original_feature_names = names

    def fit_transform(self, X, y=None):
        """Fit and transform data"""
        try:
            logging.info("Starting feature transformation with shape: {}".format(X.shape))

            # 1. Standardize original features
            X_scaled = self.scaler.fit_transform(X)
            logging.info("Standardization completed")

            # 2. Generate interaction features
            if self.interaction_method == 'gaussian':
                X_interaction = self._gaussian_interaction(X_scaled)
            else:
                raise ValueError("Unknown interaction method: {}".format(self.interaction_method))

            # 3. Post-interaction standardization - Save scaler for later use
            if self.scale_after_interaction:
                self.interaction_scaler = StandardScaler()
                X_interaction = self.interaction_scaler.fit_transform(X_interaction)
                logging.info("Post-interaction standardization completed")

            # 4. Feature selection
            if self.selection_method is not None and y is not None:
                X_final = self._select_features(X_interaction, y)
            else:
                X_final = X_interaction

            # Mark as fitted
            self.is_fitted = True

            logging.info("Final feature shape: {}".format(X_final.shape))
            if y is not None:
                logging.info("Final label shape: {}".format(y.shape))
            return X_final

        except Exception as e:
            logging.error("Error in fit_transform: {}".format(str(e)))
            raise

    # Added: transform method to apply the same transformations to new data without refitting
    def transform(self, X):
        """Transform data without refitting parameters"""
        if not self.is_fitted:
            raise ValueError("This AdvancedFeatureInteractions instance is not fitted yet. "
                             "Call 'fit_transform' before using this method.")

        try:
            logging.info("Transforming data with shape: {}".format(X.shape))

            # 1. Use the fitted scaler to transform data
            X_scaled = self.scaler.transform(X)
            logging.info("Standardization applied")

            # 2. Apply the same feature interaction
            if self.interaction_method == 'gaussian':
                X_interaction = self._gaussian_interaction(X_scaled)
            else:
                raise ValueError("Unknown interaction method: {}".format(self.interaction_method))

            # 3. Apply fitted post-interaction standardization
            if self.scale_after_interaction and self.interaction_scaler is not None:
                X_interaction = self.interaction_scaler.transform(X_interaction)
                logging.info("Post-interaction standardization applied")

            # 4. Apply existing feature selection
            if self.selection_method is not None and self.selector is not None:
                X_final = self.selector.transform(X_interaction)
                logging.info("Selected {} features".format(X_final.shape[1]))
            else:
                X_final = X_interaction

            logging.info("Final transformed shape: {}".format(X_final.shape))
            return X_final

        except Exception as e:
            logging.error("Error in transform: {}".format(str(e)))
            raise

    def _gaussian_interaction(self, X):
        """Enhanced Gaussian kernel feature interaction, adding distance features and local density features"""
        logging.info("Starting Gaussian kernel interaction...")
        n_samples, n_features = X.shape

        # Ensure feature names are available
        if not hasattr(self, 'original_feature_names') or self.original_feature_names is None:
            self.original_feature_names = ["feature_{}".format(i) for i in range(n_features)]

        # Determine gamma parameter
        if self.gaussian_gamma == 'auto':
            gamma = 1.0 / n_features
        elif isinstance(self.gaussian_gamma, (int, float)):
            gamma = self.gaussian_gamma
        else:
            gamma = 1.0 / n_features

        logging.info("Using gamma={} for Gaussian kernel".format(gamma))

        # Compute core features
        core_features = []
        core_names = []

        # 1. Direct features
        core_features.append(X)
        core_names.extend(self.original_feature_names)

        # 2. Gaussian kernel interaction for pairs of features
        for i, j in combinations(range(n_features), 2):
            # Compute Gaussian kernel between two features
            diff = X[:, i:i + 1] - X[:, j:j + 1]
            kernel = np.exp(-gamma * (diff ** 2))
            core_features.append(kernel)
            core_names.append("gaussian_{}_{}".format(self.original_feature_names[i], self.original_feature_names[j]))

            # Weighted combination
            weighted_combo = kernel * (X[:, i:i + 1] + X[:, j:j + 1])
            core_features.append(weighted_combo)
            core_names.append(
                "weighted_gaussian_{}_{}".format(self.original_feature_names[i], self.original_feature_names[j]))

        # 3. Distance features
        for i in range(n_features):
            # Compute sum of RBF kernels with all other features for the current feature
            rbf_sum = np.zeros((n_samples, 1))
            for j in range(n_features):
                if i != j:
                    diff = X[:, i:i + 1] - X[:, j:j + 1]
                    rbf_sum += np.exp(-gamma * (diff ** 2))
            core_features.append(rbf_sum)
            core_names.append("rbf_distance_{}".format(self.original_feature_names[i]))

        # 4. Local density features
        for i in range(n_features):
            # Compute local density estimate
            density = np.zeros((n_samples, 1))
            for j in range(n_features):
                diff = X[:, i:i + 1] - X[:, j:j + 1]
                density += np.exp(-gamma * np.sum(diff ** 2, axis=1, keepdims=True))
            core_features.append(density)
            core_names.append("local_density_{}".format(self.original_feature_names[i]))

        # Combine all features
        X_gaussian = np.hstack(core_features)
        self.feature_names_ = core_names

        # Log feature information
        logging.info("Created {} Gaussian kernel features:".format(len(self.feature_names_)))
        logging.info("- Original features: {}".format(n_features))
        logging.info("- Pairwise interactions: {}".format(len(list(combinations(range(n_features), 2))) * 2))
        logging.info("- Distance features: {}".format(n_features))
        logging.info("- Density features: {}".format(n_features))
        logging.info("Final feature shape: {}".format(X_gaussian.shape))

        return X_gaussian

    def _select_features(self, X, y):
        """Enhanced feature selection, supporting multiple non-linear methods"""
        # Import necessary libraries inside the method
        from sklearn.feature_selection import SelectKBest, mutual_info_classif

        logging.info("Starting feature selection with method: {}".format(self.selection_method))

        if self.selection_method == 'mutual_info':
            # Original mutual information selection
            if self.max_features is None:
                n_features = X.shape[1]
            else:
                n_features = min(self.max_features, X.shape[1])

            logging.info("Using mutual information with {} features".format(n_features))
            self.selector = SelectKBest(
                score_func=mutual_info_classif,
                k=n_features
            )
            X_selected = self.selector.fit_transform(X, y)
            self.feature_importance_ = self.selector.scores_

        elif self.selection_method == 'random_forest':
            # Random forest based feature selection
            from sklearn.ensemble import RandomForestClassifier
            if self.max_features is None:
                n_features = X.shape[1]
            else:
                n_features = min(self.max_features, X.shape[1])

            logging.info("Using random forest importance with {} features".format(n_features))
            forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            forest.fit(X, y)

            # Get feature importance
            self.feature_importance_ = forest.feature_importances_

            # Select top-k features
            indices = np.argsort(self.feature_importance_)[::-1][:n_features]

            # Create feature selection mask
            selected_mask = np.zeros(X.shape[1], dtype=bool)
            selected_mask[indices] = True

            # Implement selector to maintain API consistency
            from sklearn.feature_selection import SelectFromModel
            self.selector = SelectFromModel(forest, max_features=n_features, prefit=True)
            X_selected = self.selector.transform(X)

        elif self.selection_method == 'permutation':
            # Permutation importance based feature selection
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.inspection import permutation_importance

            if self.max_features is None:
                n_features = X.shape[1]
            else:
                n_features = min(self.max_features, X.shape[1])

            logging.info("Using permutation importance with {} features".format(n_features))

            # Train baseline algorithms
            forest = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            forest.fit(X, y)

            # Compute permutation importance
            result = permutation_importance(
                forest, X, y,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )

            # Get feature importance
            self.feature_importance_ = result.importances_mean

            # Select top-k features
            indices = np.argsort(self.feature_importance_)[::-1][:n_features]

            # Create feature selection mask
            selected_mask = np.zeros(X.shape[1], dtype=bool)
            selected_mask[indices] = True

            # Manually select features
            X_selected = X[:, selected_mask]

            # Create simple selector
            class SimpleSelector:
                def __init__(self, mask):
                    self.mask = mask

                def transform(self, X):
                    return X[:, self.mask]

                def get_support(self):
                    return self.mask

            self.selector = SimpleSelector(selected_mask)

        elif self.selection_method == 'kernel_pca':
            # Kernel PCA feature extraction
            from sklearn.decomposition import KernelPCA

            if self.max_features is None:
                n_features = min(X.shape[1], X.shape[0])
            else:
                n_features = min(self.max_features, X.shape[1], X.shape[0])

            logging.info("Using Kernel PCA with {} components".format(n_features))

            # Create Kernel PCA
            kpca = KernelPCA(
                n_components=n_features,
                kernel='rbf',
                gamma=self.gaussian_gamma if self.gaussian_gamma != 'auto' else None,
                random_state=42,
                n_jobs=-1
            )

            # Transform data
            X_selected = kpca.fit_transform(X, y)

            # Save feature importance (explained variance ratio)
            if hasattr(kpca, 'lambdas_'):
                self.feature_importance_ = kpca.lambdas_[:n_features]
            else:
                self.feature_importance_ = np.ones(n_features)

            self.selector = kpca

        elif self.selection_method == 'boruta':
            # Boruta feature selection (requires boruta_py installation)
            try:
                from boruta import BorutaPy
                from sklearn.ensemble import RandomForestClassifier

                if self.max_features is None:
                    n_features = X.shape[1]
                else:
                    n_features = min(self.max_features, X.shape[1])

                logging.info("Using Boruta feature selection")

                # Create random forest classifier
                forest = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )

                # Create Boruta feature selector
                boruta_selector = BorutaPy(
                    estimator=forest,
                    n_estimators='auto',  # Automatically select number of estimators
                    verbose=0,
                    random_state=42
                )

                # Fit data
                # Boruta requires shape [samples, features], no need to transpose if X is already in this shape
                boruta_selector.fit(X, y)

                # Select features considered important by Boruta
                X_selected = boruta_selector.transform(X)

                # If the number of selected features is less than requested, use support ranking to select more
                if X_selected.shape[1] < n_features:
                    # Select features based on Boruta ranking
                    rank_indices = np.argsort(boruta_selector.ranking_)[:n_features]
                    selected_mask = np.zeros(X.shape[1], dtype=bool)
                    selected_mask[rank_indices] = True
                    X_selected = X[:, selected_mask]

                    # Update Boruta's mask
                    boruta_mask = np.zeros(X.shape[1], dtype=bool)
                    boruta_mask[rank_indices] = True
                    boruta_selector.support_ = boruta_mask

                # Save feature importance (inverse ranking)
                self.feature_importance_ = 1.0 / boruta_selector.ranking_
                self.selector = boruta_selector

            except ImportError:
                logging.warning("Boruta not installed. Falling back to mutual_info selection.")
                # Fallback to mutual information selection
                return self._select_features_with_method(X, y, 'mutual_info')

        elif self.selection_method == 'shap':
            # SHAP value based feature selection
            try:
                import shap
                import xgboost as xgb

                if self.max_features is None:
                    n_features = X.shape[1]
                else:
                    n_features = min(self.max_features, X.shape[1])

                logging.info("Using SHAP values for feature selection with {} features".format(n_features))

                # Train algorithms
                model = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)

                # Compute SHAP values
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)

                # Compute average absolute SHAP value for each feature
                feature_importance = np.abs(shap_values.values).mean(axis=0)
                self.feature_importance_ = feature_importance

                # Select top-k features
                indices = np.argsort(self.feature_importance_)[::-1][:n_features]

                # Create feature selection mask
                selected_mask = np.zeros(X.shape[1], dtype=bool)
                selected_mask[indices] = True

                # Implement selector to maintain API consistency
                from sklearn.feature_selection import SelectFromModel
                self.selector = SelectFromModel(model, max_features=n_features, prefit=True)
                X_selected = self.selector.transform(X)

            except ImportError:
                logging.warning("SHAP not installed. Falling back to xgboost selection.")
                # Fallback to XGBoost feature selection
                return self._select_features_with_method(X, y, 'xgboost')

        elif self.selection_method == 'cmim':
            # Conditional Mutual Information Maximization
            try:
                from sklearn.feature_selection import SelectKBest

                if self.max_features is None:
                    n_features = X.shape[1]
                else:
                    n_features = min(self.max_features, X.shape[1])

                logging.info("Using CMIM with {} features".format(n_features))

                # To implement CMIM, we need to use an external library like feature_selector
                # A simplified approach based on mutual information
                from sklearn.feature_selection import mutual_info_classif

                # Get mutual information
                mi_scores = mutual_info_classif(X, y)

                # Select top-k features
                indices = np.argsort(mi_scores)[::-1][:n_features]

                # Create feature selection mask
                selected_mask = np.zeros(X.shape[1], dtype=bool)
                selected_mask[indices] = True

                # Manually select features
                X_selected = X[:, selected_mask]

                # Save feature importance
                self.feature_importance_ = mi_scores

                # Create simple selector
                class SimpleSelector:
                    def __init__(self, mask):
                        self.mask = mask

                    def transform(self, X):
                        return X[:, self.mask]

                    def get_support(self):
                        return self.mask

                self.selector = SimpleSelector(selected_mask)

            except ImportError:
                logging.warning("Required packages not installed. Falling back to mutual_info selection.")
                # Fallback to mutual information selection
                return self._select_features_with_method(X, y, 'mutual_info')

        else:
            logging.info("No feature selection or unrecognized method, keeping all features")
            X_selected = X
            self.feature_importance_ = np.ones(X.shape[1])

            # Create identity selector that does nothing
            class IdentitySelector:
                def transform(self, X):
                    return X

                def get_support(self):
                    return np.ones(X.shape[1], dtype=bool)

            self.selector = IdentitySelector()

        # Update feature names
        if len(self.feature_names_) != X.shape[1]:
            self.feature_names_ = ["feature_{}".format(i) for i in range(X.shape[1])]

        # Save selected feature names
        selected_mask = self.selector.get_support() if hasattr(self.selector, 'get_support') else np.ones(X.shape[1],
                                                                                                          dtype=bool)
        self.selected_features_ = np.array(self.feature_names_)[selected_mask]

        logging.info("Selected {} features".format(X_selected.shape[1]))
        return X_selected

    def _select_features_with_method(self, X, y, method):
        """Helper method to select features with a specified method"""
        # Save the original method
        original_method = self.selection_method

        # Set the temporary method
        self.selection_method = method

        # Call the regular select features
        X_selected = self._select_features(X, y)

        # Restore the original method
        self.selection_method = original_method

        return X_selected

    def get_feature_names(self):
        """Get feature names"""
        if self.selected_features_ is not None:
            return self.selected_features_
        return self.feature_names_

    def get_feature_importance(self):
        """Get feature importance"""
        if self.feature_importance_ is None:
            return None

        importance_df = pd.DataFrame({
            'Feature': self.feature_names_,
            'Importance': self.feature_importance_
        })
        return importance_df.sort_values('Importance', ascending=False)