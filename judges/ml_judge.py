"""
Module: judges.ml_judge
Classical ML Judge featuring Two Heads:
1. Classification Head (LogisticRegression / GradientBoosting): Predicts binary policy compliance (pass/fail).
2. Regression Head (XGBoost Regressor): Predicts continuous 1-5 quality score.
Includes GridSearch hyperparameter tuning with before/after metric logging.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
)
from scipy.stats import spearmanr

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)


class ClassicalMLJudge:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        self.classifier = None
        self.regressor = None
        self.is_trained = False
        self.tuning_log = {}

        os.makedirs(MODELS_DIR, exist_ok=True)
        self.classifier_path = os.path.join(MODELS_DIR, "ml_classifier.joblib")
        self.regressor_path = os.path.join(MODELS_DIR, "ml_regressor.joblib")
        self.vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

    @staticmethod
    def extract_features(row: Dict[str, Any]) -> Dict[str, float]:
        """Extract explicit domain feature vector from listing."""
        title = str(row.get("generated_title", ""))
        desc = str(row.get("generated_description", ""))
        brand = str(row.get("brand", "")).lower()
        bullets = row.get("source_bullets", [])
        if isinstance(bullets, list):
            bullets_text = " ".join([str(b) for b in bullets]).lower()
        else:
            bullets_text = str(bullets).lower()

        full_text = f"{title} {desc}".upper()

        # Prohibited superlatives and claims
        banned_words = [
            "BEST",
            "WORLD'S",
            "#1",
            "GUARANTEED",
            "CURE",
            "CHEAP",
            "SALE",
            "AMAZING",
            "TOP",
            "DISCOUNT",
        ]
        superlative_count = sum(1 for w in banned_words if w in full_text)

        # Structural metrics
        title_len_char = len(title)
        title_len_word = len(title.split())
        desc_len_char = len(desc)
        title_upper_ratio = sum(1 for c in title if c.isupper()) / max(
            1, title_len_char
        )
        exclamation_count = title.count("!") + desc.count("!")

        # Rule checks
        is_length_violation = 1.0 if title_len_char > 200 else 0.0
        brand_in_title = 1.0 if brand and brand in title.lower() else 0.0

        # Keyword overlap with source bullets
        title_words = set(re.findall(r"\w+", title.lower()))
        bullet_words = set(re.findall(r"\w+", bullets_text))
        overlap_count = len(title_words.intersection(bullet_words))
        overlap_ratio = overlap_count / max(1, len(title_words))

        return {
            "title_len_char": float(title_len_char),
            "title_len_word": float(title_len_word),
            "desc_len_char": float(desc_len_char),
            "superlative_count": float(superlative_count),
            "title_upper_ratio": float(title_upper_ratio),
            "exclamation_count": float(exclamation_count),
            "is_length_violation": is_length_violation,
            "brand_in_title": brand_in_title,
            "overlap_ratio": float(overlap_ratio),
        }

    def prepare_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert DataFrame into feature matrix X, compliance y_cls, and quality y_reg."""
        feature_rows = []
        texts = []
        for _, row in df.iterrows():
            feats = self.extract_features(row.to_dict())
            feature_rows.append(feats)
            texts.append(
                f"{row.get('generated_title', '')} {row.get('generated_description', '')}"
            )

        df_feats = pd.DataFrame(feature_rows)

        # Fit or transform TF-IDF
        if (
            not hasattr(self.vectorizer, "vocabulary_")
            or self.vectorizer.vocabulary_ is None
        ):
            tfidf_feats = self.vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_feats = self.vectorizer.transform(texts).toarray()

        X = np.hstack([df_feats.values, tfidf_feats])
        y_cls = (
            df["human_compliance"].astype(int).values
            if "human_compliance" in df.columns
            else np.zeros(len(df))
        )
        y_reg = (
            df["human_quality_score"].astype(float).values
            if "human_quality_score" in df.columns
            else np.zeros(len(df))
        )

        return X, y_cls, y_reg

    def train_and_tune(self, df_gold: pd.DataFrame) -> Dict[str, Any]:
        """Run GridSearch hyperparameter tuning on both heads and log performance before & after."""
        print("[INFO] Training & Hyperparameter Tuning Classical ML Judge...")
        X, y_cls, y_reg = self.prepare_dataset(df_gold)

        # -------------------------------------------------------------
        # 1. Classification Head Tuning (Policy Compliance Pass/Fail)
        # -------------------------------------------------------------
        default_cls = LogisticRegression(random_state=42, max_iter=1000)
        default_cls.fit(X, y_cls)
        y_cls_pred_init = default_cls.predict(X)
        init_f1 = f1_score(y_cls, y_cls_pred_init, zero_division=0)
        init_acc = float(np.mean(y_cls_pred_init == y_cls))

        param_grid_cls = {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]}
        grid_cls = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            param_grid_cls,
            cv=3,
            scoring="f1",
        )
        grid_cls.fit(X, y_cls)
        best_cls = grid_cls.best_estimator_
        y_cls_pred_tuned = best_cls.predict(X)
        tuned_f1 = f1_score(y_cls, y_cls_pred_tuned, zero_division=0)
        tuned_acc = float(np.mean(y_cls_pred_tuned == y_cls))

        self.classifier = best_cls

        # -------------------------------------------------------------
        # 2. Regression Head Tuning (1-5 Quality Score Prediction)
        # -------------------------------------------------------------
        if HAS_XGBOOST:
            base_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
            param_grid_reg = {
                "n_estimators": [20, 50, 100],
                "max_depth": [2, 3, 5],
                "learning_rate": [0.01, 0.1, 0.2],
            }
        else:
            from sklearn.ensemble import RandomForestRegressor

            base_reg = RandomForestRegressor(random_state=42)
            param_grid_reg = {"n_estimators": [20, 50, 100], "max_depth": [2, 4, 6]}

        base_reg.fit(X, y_reg)
        y_reg_pred_init = base_reg.predict(X)
        init_mse = mean_squared_error(y_reg, y_reg_pred_init)
        init_spearman = float(spearmanr(y_reg, y_reg_pred_init).correlation or 0.0)

        grid_reg = GridSearchCV(
            base_reg, param_grid_reg, cv=3, scoring="neg_mean_squared_error"
        )
        grid_reg.fit(X, y_reg)
        best_reg = grid_reg.best_estimator_
        y_reg_pred_tuned = best_reg.predict(X)
        tuned_mse = mean_squared_error(y_reg, y_reg_pred_tuned)
        tuned_spearman = float(spearmanr(y_reg, y_reg_pred_tuned).correlation or 0.0)

        self.regressor = best_reg
        self.is_trained = True

        # Save artifacts
        joblib.dump(self.classifier, self.classifier_path)
        joblib.dump(self.regressor, self.regressor_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)

        self.tuning_log = {
            "classification_head": {
                "model_type": "LogisticRegression",
                "default_params": {"C": 1.0, "penalty": "l2"},
                "best_params": grid_cls.best_params_,
                "before_tuning": {
                    "f1_score": round(init_f1, 4),
                    "accuracy": round(init_acc, 4),
                },
                "after_tuning": {
                    "f1_score": round(tuned_f1, 4),
                    "accuracy": round(tuned_acc, 4),
                },
            },
            "regression_head": {
                "model_type": "XGBoostRegressor"
                if HAS_XGBOOST
                else "RandomForestRegressor",
                "best_params": grid_reg.best_params_,
                "before_tuning": {
                    "mse": round(init_mse, 4),
                    "spearman": round(init_spearman, 4),
                },
                "after_tuning": {
                    "mse": round(tuned_mse, 4),
                    "spearman": round(tuned_spearman, 4),
                },
            },
        }

        print("[SUCCESS] ML Judge training complete.")
        print(f"    - Classifier F1: {init_f1:.4f} -> {tuned_f1:.4f}")
        print(f"    - Regressor MSE: {init_mse:.4f} -> {tuned_mse:.4f}")
        return self.tuning_log

    def load_models(self) -> bool:
        """Load trained models from disk if available."""
        if (
            os.path.exists(self.classifier_path)
            and os.path.exists(self.regressor_path)
            and os.path.exists(self.vectorizer_path)
        ):
            try:
                self.classifier = joblib.load(self.classifier_path)
                self.regressor = joblib.load(self.regressor_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.is_trained = True
                return True
            except Exception as e:
                print(f"[!] Error loading ML models: {e}")
        return False

    def evaluate(self, listing: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single listing using the trained ML heads (or deterministic heuristic fallback)."""
        df_single = pd.DataFrame([listing])

        if self.is_trained or self.load_models():
            X, _, _ = self.prepare_dataset(df_single)
            cls_pred = int(self.classifier.predict(X)[0])
            cls_prob = (
                float(self.classifier.predict_proba(X)[0][1])
                if hasattr(self.classifier, "predict_proba")
                else (1.0 if cls_pred == 1 else 0.0)
            )
            reg_pred = float(np.clip(self.regressor.predict(X)[0], 1.0, 5.0))

            compliance_pass = bool(cls_pred == 1)
            reasoning = f"Classical ML Judge: Predicted compliance prob={cls_prob:.2f}, predicted quality score={reg_pred:.2f}/5.0."

            return {
                "judge_name": "Classical ML Judge (LogReg + XGBoost)",
                "accuracy_score": round(reg_pred, 2),
                "compliance_pass": compliance_pass,
                "quality_score": round(reg_pred, 2),
                "reasoning": reasoning,
            }

        # Deterministic feature rule if untrained
        feats = self.extract_features(listing)
        has_banned = feats["superlative_count"] > 0
        too_long = feats["is_length_violation"] > 0
        compliance_pass = not (has_banned or too_long)

        base_score = 4.5
        base_score -= feats["superlative_count"] * 1.2
        base_score -= feats["is_length_violation"] * 1.5
        base_score += feats["overlap_ratio"] * 0.8
        quality = float(np.clip(base_score, 1.0, 5.0))

        return {
            "judge_name": "Classical ML Judge (Rule Baseline)",
            "accuracy_score": round(quality, 2),
            "compliance_pass": compliance_pass,
            "quality_score": round(quality, 2),
            "reasoning": "Untrained ML baseline: evaluated structural features & keyword overlap.",
        }


if __name__ == "__main__":
    ml_judge = ClassicalMLJudge()
    sample = {
        "brand": "BasicTech",
        "generated_title": "BasicTech Stainless Steel Knife Set",
        "generated_description": "Sharp durable blades.",
        "source_bullets": ["Stainless steel", "Kitchen knife"],
    }
    res = ml_judge.evaluate(sample)
    print("Test ML Judge Output:", res)
