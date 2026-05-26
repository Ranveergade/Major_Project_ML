# ml_engine/ml_core.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class MLEngine:
    def __init__(self):
        self.df = None
        self.target_col = None
        self.problem_type = None
        self.models_results = {}
        self.best_model = None
        self.best_score = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        
    def load_data(self, filepath):
        """Load dataset from file"""
        if filepath.endswith('.csv'):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            self.df = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            self.df = pd.read_json(filepath)
        return self.df
    
    def detect_problem_type(self, target_column):
        """Auto-detect if regression or classification"""
        self.target_col = target_column
        
        if pd.api.types.is_numeric_dtype(self.df[target_column]):
            unique_count = self.df[target_column].nunique()
            if unique_count > 20:
                self.problem_type = 'regression'
            else:
                self.problem_type = 'classification'
        else:
            self.problem_type = 'classification'
        
        return self.problem_type
    
    def clean_data(self):
        """Clean dataset - handle missing values"""
        df = self.df.copy()
        
        # Fill numeric missing with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if len(df[col].mode()) > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
        
        # Encode categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            if col != self.target_col:
                df[col] = self.le.fit_transform(df[col].astype(str))
        
        self.df = df
        return df
    
    def train_models(self, test_size=0.2, random_state=42):
        """Train multiple ML models"""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        if self.problem_type == 'classification':
            y = self.le.fit_transform(y.astype(str))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.problem_type == 'regression':
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': LinearRegression(alpha=1.0),
                'Decision Tree': DecisionTreeRegressor(random_state=random_state),
                'Random Forest': RandomForestRegressor(n_estimators=50, random_state=random_state),
                'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
                'SVR': SVR(),
            }
        else:
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
                'Decision Tree': DecisionTreeClassifier(random_state=random_state),
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=random_state),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'SVM': SVC(kernel='rbf', random_state=random_state),
            }
        
        results = {}
        
        for name, model in models.items():
            try:
                if 'SVR' in name or 'SVM' in name:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                if self.problem_type == 'regression':
                    train_score = model.score(X_train, y_train)
                    test_score = r2_score(y_test, y_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    results[name] = {
                        'train_r2': round(train_score, 4),
                        'test_r2': round(test_score, 4),
                        'test_rmse': round(test_rmse, 4)
                    }
                else:
                    train_score = model.score(X_train, y_train)
                    test_score = accuracy_score(y_test, y_pred)
                    
                    results[name] = {
                        'train_accuracy': round(train_score, 4),
                        'test_accuracy': round(test_score, 4)
                    }
                    
            except Exception as e:
                results[name] = {'error': str(e)}
        
        self.models_results = results
        
        if self.problem_type == 'regression':
            best_name = max(results, key=lambda x: results[x].get('test_r2', 0))
            self.best_score = results[best_name]['test_r2']
        else:
            best_name = max(results, key=lambda x: results[x].get('test_accuracy', 0))
            self.best_score = results[best_name]['test_accuracy']
        
        self.best_model = best_name
        
        return results
    
    def get_results(self):
        return {
            'problem_type': self.problem_type,
            'target_column': self.target_col,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'all_results': self.models_results
        }