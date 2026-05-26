# ml_engine/ml_core.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.datasets import make_classification
import warnings
import io
import base64
warnings.filterwarnings('ignore')

# Try importing XGBoost
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

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
        self.X_test = None
        self.y_test = None
        self.best_model_obj = None
        self.feature_names = []
        
    def load_data(self, filepath):
        if filepath.endswith('.csv'):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            self.df = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            self.df = pd.read_json(filepath)
        return self.df
    
    def detect_problem_type(self, target_column):
        self.target_col = target_column
        if pd.api.types.is_numeric_dtype(self.df[target_column]):
            unique_count = self.df[target_column].nunique()
            self.problem_type = 'regression' if unique_count > 20 else 'classification'
        else:
            self.problem_type = 'classification'
        return self.problem_type
    
    def clean_data(self):
        df = self.df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if len(df[col].mode()) > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
        
        for col in df.select_dtypes(include=['object']).columns:
            if col != self.target_col:
                df[col] = self.le.fit_transform(df[col].astype(str))
        
        self.feature_names = [c for c in df.columns if c != self.target_col]
        self.df = df
        return df
    
    def get_eda_summary(self):
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'numeric_cols': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': list(self.df.select_dtypes(include=['object']).columns),
            'missing': self.df.isnull().sum().sum(),
            'basic_stats': self.df.describe().to_dict()
        }
    
    def train_models(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        if self.problem_type == 'classification':
            y = self.le.fit_transform(y.astype(str))
            class_names = list(self.le.classes_)
        else:
            class_names = None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.problem_type == 'regression':
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=1.0),
                'Decision Tree': DecisionTreeRegressor(random_state=random_state, max_depth=10),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
                'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
                'SVR': SVR(),
            }
            if XGBOOST_AVAILABLE:
                models['XGBoost Regressor'] = XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0)
        else:
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
                'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=10),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'SVM': SVC(kernel='rbf', random_state=random_state, probability=True),
            }
            if XGBOOST_AVAILABLE:
                models['XGBoost Classifier'] = XGBClassifier(n_estimators=100, random_state=random_state, verbosity=0, use_label_encoder=False, eval_metric='logloss')
        
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
                    test_mae = mean_absolute_error(y_test, y_pred)
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    else:
                        importance = None
                    
                    results[name] = {
                        'train_r2': round(train_score, 4),
                        'test_r2': round(test_score, 4),
                        'test_rmse': round(test_rmse, 4),
                        'test_mae': round(test_mae, 4),
                        'feature_importance': importance
                    }
                else:
                    train_score = model.score(X_train, y_train)
                    test_score = accuracy_score(y_test, y_pred)
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    cr = classification_report(y_test, y_pred, output_dict=True)
                    
                    # ROC curve data (for binary classification)
                    roc_data = None
                    if hasattr(model, 'predict_proba') and len(class_names) == 2:
                        try:
                            y_proba = model.predict_proba(X_test_scaled if 'SVM' in name else X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_proba)
                            roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': round(auc(fpr, tpr), 4)}
                        except:
                            pass
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    else:
                        importance = None
                    
                    results[name] = {
                        'train_accuracy': round(train_score, 4),
                        'test_accuracy': round(test_score, 4),
                        'confusion_matrix': cm.tolist(),
                        'classification_report': cr,
                        'roc_data': roc_data,
                        'feature_importance': importance,
                        'class_names': class_names
                    }
                    
            except Exception as e:
                results[name] = {'error': str(e)}
        
        self.models_results = results
        
        # Find best model
        if self.problem_type == 'regression':
            best_name = max(results, key=lambda x: results[x].get('test_r2', 0))
            self.best_score = results[best_name]['test_r2']
        else:
            best_name = max(results, key=lambda x: results[x].get('test_accuracy', 0))
            self.best_score = results[best_name]['test_accuracy']
        
        self.best_model = best_name
        self.best_model_obj = models[best_name]
        
        return results
    
    def create_visualizations(self):
        """Create visualizations for the best model"""
        visualizations = {}
        
        # 1. Confusion Matrix Heatmap
        if self.problem_type == 'classification' and self.best_model_obj is not None:
            try:
                if 'SVM' in self.best_model or 'SVR' in self.best_model:
                    y_pred = self.best_model_obj.predict(self.scaler.transform(self.X_test))
                else:
                    y_pred = self.best_model_obj.predict(self.X_test)
                
                cm = confusion_matrix(self.y_test, y_pred)
                
                # Create confusion matrix plot
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                          xticklabels=self.le.classes_,
                          yticklabels=self.le.classes_)
                plt.title(f'Confusion Matrix - {self.best_model}', fontsize=14, fontweight='bold')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['confusion_matrix'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except Exception as e:
                visualizations['confusion_matrix'] = None
        
        # 2. Feature Importance
        if self.best_model_obj is not None and hasattr(self.best_model_obj, 'feature_importances_'):
            try:
                importance = self.best_model_obj.feature_importances_
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=True).tail(15)
                
                plt.figure(figsize=(10, 8))
                plt.barh(feature_imp['feature'], feature_imp['importance'], color='steelblue')
                plt.xlabel('Importance')
                plt.title(f'Feature Importance - {self.best_model}', fontsize=14, fontweight='bold')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
            except:
                visualizations['feature_importance'] = None
        
        # 3. Classification Report Bar Chart
        if self.problem_type == 'classification':
            best_result = self.models_results.get(self.best_model, {})
            if 'classification_report' in best_result:
                try:
                    cr = best_result['classification_report']
                    metrics_df = pd.DataFrame({
                        'Precision': [cr.get(c, {}).get('precision', 0) for c in cr.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']],
                        'Recall': [cr.get(c, {}).get('recall', 0) for c in cr.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']],
                        'F1-Score': [cr.get(c, {}).get('f1-score', 0) for c in cr.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
                    }, index=[c for c in cr.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']])
                    
                    if not metrics_df.empty:
                        plt.figure(figsize=(10, 6))
                        metrics_df.plot(kind='bar', colormap='viridis')
                        plt.title('Classification Report', fontsize=14, fontweight='bold')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        plt.close()
                        buf.seek(0)
                        visualizations['classification_report'] = base64.b64encode(buf.read()).decode('utf-8')
                        buf.close()
                except:
                    visualizations['classification_report'] = None
        
        # 4. Model Comparison Bar Chart
        try:
            if self.problem_type == 'regression':
                model_scores = {k: v.get('test_r2', 0) for k, v in self.models_results.items() if 'test_r2' in v}
            else:
                model_scores = {k: v.get('test_accuracy', 0) for k, v in self.models_results.items() if 'test_accuracy' in v}
            
            if model_scores:
                plt.figure(figsize=(12, 6))
                colors = ['#00d4aa' if k == self.best_model else '#667eea' for k in model_scores.keys()]
                plt.bar(model_scores.keys(), model_scores.values(), color=colors)
                plt.title('Model Comparison', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['model_comparison'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
        except:
            visualizations['model_comparison'] = None
        
        # 5. Correlation Heatmap
        try:
            numeric_df = self.df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                visualizations['correlation_heatmap'] = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
        except:
            visualizations['correlation_heatmap'] = None
        
        return visualizations
    
    def get_results(self):
        return {
            'problem_type': self.problem_type,
            'target_column': self.target_col,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'all_results': self.models_results
        }