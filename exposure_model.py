import argparse
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ExposureModel")

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class Position:
    asset_id: int
    asset_name: str
    asset_class: str
    position_type: str
    notional: float
    direction: int  # 1 = long, -1 = short
    
    def __post_init__(self):
        if self.direction not in [-1, 1]:
            raise ValueError(f"Direction must be 1 or -1, got {self.direction}")


@dataclass
class CSATerms:
    
    has_csa: bool
    threshold: float
    minimum_transfer_amount: float
    margin_period_of_risk_days: int
    independent_amount: float
    rehypothecation_allowed: bool
    
    @property
    def mpor_years(self) -> float:
        return self.margin_period_of_risk_days / 252.0


@dataclass
class Counterparty:
    name: str
    credit_rating: str
    cds_spread_bps: float
    recovery_rate: float
    lgd: float
    
    @property
    def hazard_rate(self) -> float:
        return (self.cds_spread_bps / 10000) / self.lgd


@dataclass
class SimulationConfig:
    num_simulations: int
    horizon_days: int
    time_steps: int
    seed: int
    use_antithetic: bool
    confidence_level: float
    
    @property
    def dt(self) -> float:
        return (self.horizon_days / 252.0) / self.time_steps


@dataclass
class MarketParams:
    asset_names: List[str]
    spot_prices: np.ndarray
    drift: np.ndarray
    volatility: np.ndarray
    correlation: np.ndarray
    cholesky_matrix: np.ndarray
    estimation_date: str
    estimation_window: int


@dataclass
class ExposureProfile:
    time_grid: np.ndarray  # in years
    time_grid_days: np.ndarray
    ee: np.ndarray
    ee_lower: np.ndarray
    ee_upper: np.ndarray
    pfe: Dict[float, np.ndarray]
    effective_epe: float
    peak_exposure: float
    peak_exposure_time: float
    expected_positive_exposure: float
    collateralized_ee: Optional[np.ndarray] = None
    collateralized_pfe: Optional[Dict[float, np.ndarray]] = None


@dataclass
class CVAResult:
    cva_value: float
    cva_as_percentage: float
    marginal_cva: np.ndarray
    survival_probabilities: np.ndarray
    discount_factors: np.ndarray
    expected_loss_profile: np.ndarray


@dataclass 
class WrongWayRiskMetrics:
    wwr_correlation: float
    wwr_adjustment_factor: float
    stressed_ee: np.ndarray
    wwr_category: str  # 'LOW', 'MEDIUM', 'HIGH'

class MarketDataEstimator:
    
    TRADING_DAYS_PER_YEAR = 252
    
    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate
        
    def estimate_from_prices(
        self, 
        price_df: pd.DataFrame,
        use_risk_neutral_drift: bool = True
    ) -> MarketParams:
        logger.info("Estimating market parameters from historical data")
        
        # Validate input
        if price_df.empty:
            raise ValueError("Price DataFrame is empty")
            
        # Calculate log returns
        log_returns = np.log(price_df / price_df.shift(1)).dropna()
        
        if len(log_returns) < 30:
            logger.warning(f"Only {len(log_returns)} observations - estimates may be unreliable")
        
        # Annualized statistics
        daily_mean = log_returns.mean()
        daily_std = log_returns.std(ddof=1)
        
        # Historical drift and volatility
        historical_drift = daily_mean * self.TRADING_DAYS_PER_YEAR
        volatility = daily_std * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        # Use risk-neutral drift if specified (for pricing)
        if use_risk_neutral_drift:
            drift = np.full_like(volatility, self.risk_free_rate)
            logger.info("Using risk-neutral drift (r = %.4f)", self.risk_free_rate)
        else:
            drift = historical_drift.values
            logger.info("Using historical drift estimates")
        
        # Correlation matrix
        correlation = log_returns.corr().values
        
        # Validate correlation matrix
        if not self._is_valid_correlation(correlation):
            logger.warning("Correlation matrix not positive definite - applying correction")
            correlation = self._nearest_positive_definite(correlation)
        
        # Cholesky decomposition
        cholesky_matrix = self._safe_cholesky(correlation)
        
        # Spot prices (most recent)
        spot_prices = price_df.iloc[-1].values.astype(float)
        
        params = MarketParams(
            asset_names=price_df.columns.tolist(),
            spot_prices=spot_prices,
            drift=drift if isinstance(drift, np.ndarray) else drift.values,
            volatility=volatility.values,
            correlation=correlation,
            cholesky_matrix=cholesky_matrix,
            estimation_date=str(price_df.index[-1].date()),
            estimation_window=len(log_returns)
        )
        
        # Log summary statistics
        logger.info(f"Estimation window: {params.estimation_window} days")
        logger.info(f"Assets: {params.asset_names}")
        logger.info(f"Spot prices: {dict(zip(params.asset_names, params.spot_prices))}")
        logger.info(f"Annualized volatilities: {dict(zip(params.asset_names, np.round(params.volatility, 4)))}")
        
        return params
    
    @staticmethod
    def _is_valid_correlation(corr: np.ndarray, tol: float = 1e-8) -> bool:
        eigenvalues = np.linalg.eigvalsh(corr)
        return np.all(eigenvalues >= -tol)
    
    @staticmethod
    def _nearest_positive_definite(A: np.ndarray) -> np.ndarray:
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        if MarketDataEstimator._is_valid_correlation(A3):
            return A3
            
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not MarketDataEstimator._is_valid_correlation(A3):
            min_eig = np.min(np.real(np.linalg.eigvalsh(A3)))
            A3 += I * (-min_eig * k**2 + spacing)
            k += 1
        return A3
    
    @staticmethod
    def _safe_cholesky(corr: np.ndarray) -> np.ndarray:
        eps = 1e-10
        n = corr.shape[0]
        
        # Ensure diagonal is exactly 1
        corr = corr.copy()
        np.fill_diagonal(corr, 1.0)
        
        # Add small regularization if needed
        try:
            L = cholesky(corr, lower=True)
        except np.linalg.LinAlgError:
            corr += eps * np.eye(n)
            L = cholesky(corr, lower=True)
            
        return L

class MonteCarloEngine:
    def __init__(self, config: SimulationConfig, market_params: MarketParams):
        self.config = config
        self.params = market_params
        self.rng = np.random.Generator(np.random.PCG64(config.seed))
        
    def simulate_paths(self, show_progress: bool = True) -> np.ndarray:
        n_sims = self.config.num_simulations
        n_steps = self.config.time_steps
        n_assets = len(self.params.spot_prices)
        dt = self.config.dt
        
        logger.info(f"Starting simulation: {n_sims:,} paths, {n_steps} steps, {n_assets} assets")
        
        if self.config.use_antithetic:
            return self._simulate_antithetic(n_sims, n_steps, n_assets, dt, show_progress)
        else:
            return self._simulate_standard(n_sims, n_steps, n_assets, dt, show_progress)
    
    def _simulate_standard(
        self, 
        n_sims: int, 
        n_steps: int, 
        n_assets: int, 
        dt: float,
        show_progress: bool
    ) -> np.ndarray:
        paths = np.empty((n_sims, n_steps + 1, n_assets), dtype=np.float64)
        paths[:, 0, :] = self.params.spot_prices
        
        sqrt_dt = np.sqrt(dt)
        drift_term = (self.params.drift - 0.5 * self.params.volatility**2) * dt
        vol_term = self.params.volatility * sqrt_dt
        L = self.params.cholesky_matrix
        
        iterator = tqdm(range(1, n_steps + 1), desc="Simulating", disable=not show_progress)
        
        for t in iterator:
            Z = self.rng.standard_normal((n_sims, n_assets))
            correlated_Z = Z @ L.T
            
            log_return = drift_term + vol_term * correlated_Z
            paths[:, t, :] = paths[:, t-1, :] * np.exp(log_return)
            
        return paths
    
    def _simulate_antithetic(
        self, 
        n_sims: int, 
        n_steps: int, 
        n_assets: int, 
        dt: float,
        show_progress: bool
    ) -> np.ndarray:
        half_sims = n_sims // 2
        paths = np.empty((n_sims, n_steps + 1, n_assets), dtype=np.float64)
        paths[:, 0, :] = self.params.spot_prices
        
        sqrt_dt = np.sqrt(dt)
        drift_term = (self.params.drift - 0.5 * self.params.volatility**2) * dt
        vol_term = self.params.volatility * sqrt_dt
        L = self.params.cholesky_matrix
        
        iterator = tqdm(range(1, n_steps + 1), desc="Simulating (antithetic)", disable=not show_progress)
        
        for t in iterator:
            Z = self.rng.standard_normal((half_sims, n_assets))
            
            # Regular paths
            correlated_Z = Z @ L.T
            log_return = drift_term + vol_term * correlated_Z
            paths[:half_sims, t, :] = paths[:half_sims, t-1, :] * np.exp(log_return)
            
            # Antithetic paths
            correlated_Z_anti = -Z @ L.T
            log_return_anti = drift_term + vol_term * correlated_Z_anti
            paths[half_sims:, t, :] = paths[half_sims:, t-1, :] * np.exp(log_return_anti)
            
        return paths
    
    def compute_simulation_diagnostics(self, paths: np.ndarray) -> pd.DataFrame:
        n_assets = paths.shape[2]
        T = self.config.horizon_days / 252
        
        diagnostics = []
        
        for i in range(n_assets):
            terminal_prices = paths[:, -1, i]
            log_returns = np.log(terminal_prices / paths[:, 0, i])
            
            realized_mean = log_returns.mean() / T
            realized_vol = log_returns.std() / np.sqrt(T)
            
            expected_mean = self.params.drift[i] - 0.5 * self.params.volatility[i]**2
            expected_vol = self.params.volatility[i]
            
            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = stats.jarque_bera(log_returns)
            
            diagnostics.append({
                'asset': self.params.asset_names[i],
                'expected_drift': expected_mean,
                'realized_drift': realized_mean,
                'drift_error_pct': abs(realized_mean - expected_mean) / abs(expected_mean) * 100 if expected_mean != 0 else 0,
                'expected_vol': expected_vol,
                'realized_vol': realized_vol,
                'vol_error_pct': abs(realized_vol - expected_vol) / expected_vol * 100,
                'terminal_mean': terminal_prices.mean(),
                'terminal_std': terminal_prices.std(),
                'jb_statistic': jb_stat,
                'jb_pvalue': jb_pvalue,
                'normality_ok': jb_pvalue > 0.05
            })
            
        return pd.DataFrame(diagnostics)

class ExposureCalculator:    
    def __init__(
        self, 
        positions: List[Position],
        csa_terms: Optional[CSATerms],
        risk_free_rate: float
    ):
        self.positions = positions
        self.csa = csa_terms
        self.risk_free_rate = risk_free_rate
        
    def compute_portfolio_values(
        self, 
        paths: np.ndarray,
        spot_prices: np.ndarray
    ) -> np.ndarray:
        n_sims, n_times, n_assets = paths.shape
        
        # Build position vector
        position_weights = np.zeros(n_assets)
        for pos in self.positions:
            idx = pos.asset_id - 1  # Convert to 0-based
            # Notional in units of the asset
            units = pos.notional / spot_prices[idx]
            position_weights[idx] = units * pos.direction
        
        # Compute portfolio value: sum of (units × price change)
        price_changes = paths - spot_prices  # (n_sims, n_times, n_assets)
        portfolio_values = np.einsum('ijk,k->ij', price_changes, position_weights)
        
        return portfolio_values
    
    def compute_exposure_profile(
        self,
        portfolio_values: np.ndarray,
        time_grid_years: np.ndarray,
        pfe_percentiles: List[float],
        confidence_level: float = 0.95
    ) -> ExposureProfile:
        logger.info("Computing exposure profile")
        
        # Positive exposures only (exposure = max(V, 0))
        exposures = np.maximum(portfolio_values, 0)
        
        n_sims, n_times = exposures.shape
        
        # Expected Exposure
        ee = exposures.mean(axis=0)
        
        # Standard error and confidence intervals
        ee_std = exposures.std(axis=0, ddof=1) / np.sqrt(n_sims)
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        ee_lower = ee - z_score * ee_std
        ee_upper = ee + z_score * ee_std
        
        # PFE at various percentiles
        pfe = {}
        for p in pfe_percentiles:
            pfe[p] = np.percentile(exposures, p, axis=0)
        
        # Effective EPE (Basel III definition: time-averaged maximum of EE)
        cummax_ee = np.maximum.accumulate(ee)
        dt = time_grid_years[1] - time_grid_years[0] if len(time_grid_years) > 1 else time_grid_years[0]
        effective_epe = np.trapz(cummax_ee, time_grid_years) / time_grid_years[-1]
        
        # Peak exposure
        peak_idx = np.argmax(ee)
        peak_exposure = ee[peak_idx]
        peak_time = time_grid_years[peak_idx]
        
        # Expected Positive Exposure (simple average)
        epe = np.trapz(ee, time_grid_years) / time_grid_years[-1]
        
        logger.info(f"Peak EE: ${peak_exposure:,.0f} at t={peak_time:.2f}y")
        logger.info(f"Effective EPE: ${effective_epe:,.0f}")
        
        return ExposureProfile(
            time_grid=time_grid_years,
            time_grid_days=(time_grid_years * 252).astype(int),
            ee=ee,
            ee_lower=ee_lower,
            ee_upper=ee_upper,
            pfe=pfe,
            effective_epe=effective_epe,
            peak_exposure=peak_exposure,
            peak_exposure_time=peak_time,
            expected_positive_exposure=epe
        )
    
    def apply_collateral(
        self,
        exposures: np.ndarray,
        time_grid_years: np.ndarray
    ) -> np.ndarray:
        if self.csa is None or not self.csa.has_csa:
            return exposures
            
        logger.info("Applying CSA collateral")
        
        n_sims, n_times = exposures.shape
        collateralized = np.zeros_like(exposures)
        
        threshold = self.csa.threshold
        mta = self.csa.minimum_transfer_amount
        ia = self.csa.independent_amount
        mpor_steps = max(1, int(self.csa.margin_period_of_risk_days / 252 / (time_grid_years[1] - time_grid_years[0])))
        
        for t in range(n_times):
            if t < mpor_steps:
                collateralized[:, t] = exposures[:, t]
            else:
                # Collateral based on exposure at t-MPOR
                exposure_at_call = exposures[:, t - mpor_steps]
                collateral_required = np.maximum(exposure_at_call - threshold - ia, 0)
                # Apply MTA: only transfer if above MTA
                collateral_posted = np.where(collateral_required >= mta, collateral_required, 0)
                collateralized[:, t] = np.maximum(exposures[:, t] - collateral_posted, 0)
        
        return collateralized

class CVACalculator:    
    def __init__(
        self, 
        counterparty: Counterparty, 
        risk_free_rate: float
    ):
        self.counterparty = counterparty
        self.risk_free_rate = risk_free_rate
        
    def compute_cva(
        self,
        exposure_profile: ExposureProfile
    ) -> CVAResult:
        logger.info(f"Computing CVA for {self.counterparty.name}")
        
        time_grid = exposure_profile.time_grid
        ee = exposure_profile.ee
        n_times = len(time_grid)
        
        # Discount factors
        discount_factors = np.exp(-self.risk_free_rate * time_grid)
        
        # Survival probabilities: Q(t) = exp(-λt)
        hazard_rate = self.counterparty.hazard_rate
        survival_prob = np.exp(-hazard_rate * time_grid)
        
        # Default probabilities in each interval: Q(t_{i-1}) - Q(t_i)
        default_prob = np.zeros(n_times)
        default_prob[0] = 1 - survival_prob[0]
        default_prob[1:] = survival_prob[:-1] - survival_prob[1:]
        
        # Expected loss profile
        expected_loss = self.counterparty.lgd * ee * discount_factors * default_prob
        
        # Total CVA
        cva = expected_loss.sum()
        
        # CVA as percentage of initial exposure
        initial_exposure = ee[0] if ee[0] > 0 else exposure_profile.peak_exposure
        cva_pct = (cva / initial_exposure * 100) if initial_exposure > 0 else 0
        
        logger.info(f"CVA: ${cva:,.0f} ({cva_pct:.2f}% of exposure)")
        logger.info(f"Implied hazard rate: {hazard_rate*100:.2f}%")
        logger.info(f"1Y survival probability: {survival_prob[min(3, n_times-1)]:.4f}")
        
        return CVAResult(
            cva_value=cva,
            cva_as_percentage=cva_pct,
            marginal_cva=expected_loss,
            survival_probabilities=survival_prob,
            discount_factors=discount_factors,
            expected_loss_profile=expected_loss
        )

class WrongWayRiskAnalyzer:
    
    # WWR correlation thresholds (industry practice)
    WWR_THRESHOLDS = {
        'LOW': 0.2,
        'MEDIUM': 0.5,
        'HIGH': 0.8
    }
    
    def __init__(self, counterparty: Counterparty):
        self.counterparty = counterparty
        
    def analyze_wwr(
        self,
        portfolio_values: np.ndarray,
        simulated_credit_index: Optional[np.ndarray] = None
    ) -> WrongWayRiskMetrics:
        logger.info("Analyzing Wrong-Way Risk")
        
        n_sims, n_times = portfolio_values.shape
        
        # Use portfolio losses as credit proxy (simplified approach)
        if simulated_credit_index is None:
            # Use terminal portfolio values as proxy
            credit_proxy = -portfolio_values[:, -1]  # Losses as credit stress
        else:
            credit_proxy = simulated_credit_index
            
        terminal_exposure = np.maximum(portfolio_values[:, -1], 0)
        
        # Correlation between exposure and credit deterioration
        wwr_corr = np.corrcoef(terminal_exposure, credit_proxy)[0, 1]
        
        # Categorize WWR
        abs_corr = abs(wwr_corr)
        if abs_corr < self.WWR_THRESHOLDS['LOW']:
            wwr_category = 'LOW'
        elif abs_corr < self.WWR_THRESHOLDS['MEDIUM']:
            wwr_category = 'MEDIUM'
        else:
            wwr_category = 'HIGH'
        
        # WWR adjustment factor (simplified Hull-White approach)
        # α = 1 + ρ × σ_credit/σ_exposure × CVA_factor
        wwr_adjustment = 1 + max(0, wwr_corr) * 0.2  # Simplified multiplier
        
        # Stressed EE: condition on credit stress (top decile of credit proxy)
        stress_threshold = np.percentile(credit_proxy, 90)
        stressed_mask = credit_proxy >= stress_threshold
        stressed_ee = np.maximum(portfolio_values[stressed_mask, :], 0).mean(axis=0)
        
        logger.info(f"WWR correlation: {wwr_corr:.3f}")
        logger.info(f"WWR category: {wwr_category}")
        logger.info(f"WWR adjustment factor: {wwr_adjustment:.3f}")
        
        return WrongWayRiskMetrics(
            wwr_correlation=wwr_corr,
            wwr_adjustment_factor=wwr_adjustment,
            stressed_ee=stressed_ee,
            wwr_category=wwr_category
        )

class ReportGenerator:    
    def __init__(self, output_dir: str, plot_format: str = 'png', plot_dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
        
    def save_exposure_profile(
        self, 
        profile: ExposureProfile,
        filename: str = "exposure_profile.csv"
    ) -> str:
        df = pd.DataFrame({
            'time_step': range(len(profile.time_grid)),
            'time_years': profile.time_grid,
            'time_days': profile.time_grid_days,
            'expected_exposure': profile.ee,
            'ee_lower_ci': profile.ee_lower,
            'ee_upper_ci': profile.ee_upper
        })
        
        for p, values in profile.pfe.items():
            df[f'pfe_{int(p)}'] = values
            
        if profile.collateralized_ee is not None:
            df['ee_collateralized'] = profile.collateralized_ee
            
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False, float_format='%.2f')
        logger.info(f"Saved exposure profile: {filepath}")
        return str(filepath)
    
    def save_cva_summary(
        self,
        cva_result: CVAResult,
        counterparty: Counterparty,
        filename: str = "cva_summary.csv"
    ) -> str:
        summary = {
            'counterparty': counterparty.name,
            'credit_rating': counterparty.credit_rating,
            'cds_spread_bps': counterparty.cds_spread_bps,
            'recovery_rate': counterparty.recovery_rate,
            'lgd': counterparty.lgd,
            'hazard_rate': counterparty.hazard_rate,
            'cva_value': cva_result.cva_value,
            'cva_percentage': cva_result.cva_as_percentage,
            '1y_survival_prob': cva_result.survival_probabilities[min(3, len(cva_result.survival_probabilities)-1)]
        }
        
        df = pd.DataFrame([summary])
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved CVA summary: {filepath}")
        return str(filepath)
    
    def save_simulation_diagnostics(
        self,
        diagnostics: pd.DataFrame,
        filename: str = "simulation_diagnostics.csv"
    ) -> str:
        filepath = self.output_dir / filename
        diagnostics.to_csv(filepath, index=False, float_format='%.6f')
        logger.info(f"Saved simulation diagnostics: {filepath}")
        return str(filepath)
    
    def plot_exposure_profile(
        self,
        profile: ExposureProfile,
        cva_result: Optional[CVAResult] = None,
        wwr_metrics: Optional[WrongWayRiskMetrics] = None,
        filename: str = "exposure_profile.png"
    ) -> str:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        time_grid = profile.time_grid
        
        # Plot 1: EE and PFE
        ax1 = axes[0, 0]
        ax1.fill_between(time_grid, profile.ee_lower, profile.ee_upper, 
                         alpha=0.3, color='blue', label='95% CI')
        ax1.plot(time_grid, profile.ee, 'b-', linewidth=2, label='EE')
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(profile.pfe)))
        for (p, values), color in zip(sorted(profile.pfe.items()), colors):
            ax1.plot(time_grid, values, '--', color=color, linewidth=1.5, label=f'PFE {int(p)}%')
        
        if profile.collateralized_ee is not None:
            ax1.plot(time_grid, profile.collateralized_ee, 'g-', linewidth=2, 
                    label='EE (Collateralized)')
        
        ax1.axhline(y=profile.effective_epe, color='orange', linestyle=':', 
                   linewidth=1.5, label=f'Effective EPE: ${profile.effective_epe:,.0f}')
        ax1.scatter([profile.peak_exposure_time], [profile.peak_exposure], 
                   color='red', s=100, zorder=5, label=f'Peak: ${profile.peak_exposure:,.0f}')
        
        ax1.set_xlabel('Time (years)')
        ax1.set_ylabel('Exposure ($)')
        ax1.set_title('Expected Exposure & Potential Future Exposure Profile')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        # Plot 2: PFE term structure
        ax2 = axes[0, 1]
        pfe_percentiles = sorted(profile.pfe.keys())
        for t_idx, t in enumerate([0.25, 0.5, 0.75, 1.0]):
            t_step = int(t * len(time_grid) / time_grid[-1]) if time_grid[-1] > 0 else 0
            t_step = min(t_step, len(time_grid) - 1)
            pfe_values = [profile.pfe[p][t_step] for p in pfe_percentiles]
            ax2.plot(pfe_percentiles, pfe_values, 'o-', label=f't={t}y')
        
        ax2.set_xlabel('Percentile')
        ax2.set_ylabel('PFE ($)')
        ax2.set_title('PFE Term Structure by Percentile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        # Plot 3: CVA contribution
        ax3 = axes[1, 0]
        if cva_result is not None:
            ax3.bar(time_grid, cva_result.expected_loss_profile, width=time_grid[1]-time_grid[0] if len(time_grid) > 1 else 0.01,
                   alpha=0.7, color='red', label='Marginal CVA')
            ax3.set_xlabel('Time (years)')
            ax3.set_ylabel('Expected Loss ($)')
            ax3.set_title(f'CVA Contribution by Time Period (Total: ${cva_result.cva_value:,.0f})')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add survival probability on secondary axis
            ax3_twin = ax3.twinx()
            ax3_twin.plot(time_grid, cva_result.survival_probabilities, 'g--', 
                         linewidth=1.5, label='Survival Prob')
            ax3_twin.set_ylabel('Survival Probability', color='green')
            ax3_twin.tick_params(axis='y', labelcolor='green')
        else:
            ax3.text(0.5, 0.5, 'CVA not calculated', ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Wrong-Way Risk
        ax4 = axes[1, 1]
        if wwr_metrics is not None:
            ax4.plot(time_grid, profile.ee, 'b-', linewidth=2, label='EE (Base)')
            ax4.plot(time_grid, wwr_metrics.stressed_ee, 'r-', linewidth=2, label='EE (Stressed)')
            ax4.fill_between(time_grid, profile.ee, wwr_metrics.stressed_ee, 
                           alpha=0.3, color='red', label='WWR Impact')
            ax4.set_xlabel('Time (years)')
            ax4.set_ylabel('Exposure ($)')
            ax4.set_title(f'Wrong-Way Risk Analysis (WWR Category: {wwr_metrics.wwr_category})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
            
            # Add WWR metrics annotation
            ax4.annotate(f'ρ = {wwr_metrics.wwr_correlation:.3f}\nAdj = {wwr_metrics.wwr_adjustment_factor:.3f}',
                        xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax4.text(0.5, 0.5, 'WWR not analyzed', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved exposure plot: {filepath}")
        return str(filepath)
    
    def generate_summary_json(
        self,
        profile: ExposureProfile,
        cva_result: Optional[CVAResult],
        wwr_metrics: Optional[WrongWayRiskMetrics],
        market_params: MarketParams,
        sim_config: SimulationConfig,
        filename: str = "exposure_summary.json"
    ) -> str:
        summary = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model_version": "2.1.0",
                "regulatory_reference": "Basel III SA-CCR (BCBS 279)"
            },
            "simulation_config": {
                "num_simulations": sim_config.num_simulations,
                "horizon_days": sim_config.horizon_days,
                "time_steps": sim_config.time_steps,
                "antithetic_variates": sim_config.use_antithetic
            },
            "market_parameters": {
                "estimation_date": market_params.estimation_date,
                "estimation_window_days": market_params.estimation_window,
                "assets": market_params.asset_names,
                "spot_prices": {k: float(v) for k, v in zip(market_params.asset_names, market_params.spot_prices)},
                "volatilities": {k: float(v) for k, v in zip(market_params.asset_names, market_params.volatility)}
            },
            "exposure_metrics": {
                "peak_exposure": float(profile.peak_exposure),
                "peak_exposure_time_years": float(profile.peak_exposure_time),
                "effective_epe": float(profile.effective_epe),
                "expected_positive_exposure": float(profile.expected_positive_exposure),
                "terminal_ee": float(profile.ee[-1]),
                "pfe_95_1y": float(profile.pfe[95][-1]) if 95 in profile.pfe else None,
                "pfe_99_1y": float(profile.pfe[99][-1]) if 99 in profile.pfe else None
            }
        }
        
        if cva_result:
            summary["cva_metrics"] = {
                "cva_value": float(cva_result.cva_value),
                "cva_percentage": float(cva_result.cva_as_percentage),
                "1y_survival_probability": float(cva_result.survival_probabilities[min(12, len(cva_result.survival_probabilities)-1)])
            }
            
        if wwr_metrics:
            summary["wrong_way_risk"] = {
                "correlation": float(wwr_metrics.wwr_correlation),
                "adjustment_factor": float(wwr_metrics.wwr_adjustment_factor),
                "category": wwr_metrics.wwr_category
            }
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary JSON: {filepath}")
        return str(filepath)

def generate_sample_historical_data(
    assets: List[str] = ["SPX", "EURUSD", "UST10Y"],
    days: int = 504,
    seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)
    
    # Initial prices and parameters
    params = {
        "SPX": {"S0": 4500, "mu": 0.08, "sigma": 0.18},
        "EURUSD": {"S0": 1.08, "mu": 0.00, "sigma": 0.08},
        "UST10Y": {"S0": 95, "mu": 0.02, "sigma": 0.06}
    }
    
    # Correlation matrix (realistic estimates)
    corr_matrix = np.array([
        [1.0, 0.3, -0.4],
        [0.3, 1.0, 0.1],
        [-0.4, 0.1, 1.0]
    ])
    
    L = cholesky(corr_matrix, lower=True)
    dt = 1/252
    
    prices = {asset: [params[asset]["S0"]] for asset in assets}
    
    for _ in range(1, days):
        Z = np.random.standard_normal(len(assets))
        corr_Z = Z @ L.T
        
        for i, asset in enumerate(assets):
            p = params[asset]
            log_ret = (p["mu"] - 0.5 * p["sigma"]**2) * dt + p["sigma"] * np.sqrt(dt) * corr_Z[i]
            prices[asset].append(prices[asset][-1] * np.exp(log_ret))
    
    # Create DataFrame with date index
    dates = pd.bdate_range(end=datetime.now(), periods=days)
    df = pd.DataFrame(prices, index=dates)
    df.index.name = 'date'
    
    return df

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Counterparty Credit Exposure Model (EE/PFE/CVA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python counterparty_exposure_model.py --config exposure_config.yaml
  python counterparty_exposure_model.py --generate-sample --sample-output sample_prices.csv
  
References:
  - Basel III SA-CCR: BCBS 279 (March 2014)
  - Gregory (2015): The xVA Challenge
        """
    )
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--generate-sample", action="store_true", 
                       help="Generate sample historical price data")
    parser.add_argument("--sample-output", type=str, default="historical_prices.csv",
                       help="Output path for sample data")
    
    args = parser.parse_args()
    
    if args.generate_sample:
        logger.info("Generating sample historical price data")
        df = generate_sample_historical_data()
        df.to_csv(args.sample_output)
        logger.info(f"Sample data saved to {args.sample_output}")
        return
    
    if not args.config:
        parser.print_help()
        logger.error("Please provide --config or use --generate-sample")
        sys.exit(1)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Initialize components
    sim_config = SimulationConfig(
        num_simulations=config['simulation']['num_simulations'],
        horizon_days=config['simulation']['horizon_days'],
        time_steps=config['simulation']['time_steps'],
        seed=config['simulation']['seed'],
        use_antithetic=config['simulation']['use_antithetic'],
        confidence_level=config['simulation']['confidence_level']
    )
    
    counterparty = Counterparty(
        name=config['counterparty']['name'],
        credit_rating=config['counterparty']['credit_rating'],
        cds_spread_bps=config['counterparty']['cds_spread_bps'],
        recovery_rate=config['counterparty']['recovery_rate'],
        lgd=config['counterparty']['lgd']
    )
    
    csa_terms = None
    if config['netting_set']['has_csa']:
        csa_terms = CSATerms(
            has_csa=True,
            threshold=config['netting_set']['threshold'],
            minimum_transfer_amount=config['netting_set']['minimum_transfer_amount'],
            margin_period_of_risk_days=config['netting_set']['margin_period_of_risk_days'],
            independent_amount=config['netting_set']['independent_amount'],
            rehypothecation_allowed=config['netting_set']['rehypothecation_allowed']
        )
    
    positions = [
        Position(
            asset_id=p['asset_id'],
            asset_name=p['asset_name'],
            asset_class=p['asset_class'],
            position_type=p['position_type'],
            notional=p['notional'],
            direction=p['direction']
        )
        for p in config['positions']
    ]
    
    # Load historical prices
    price_file = config['market_data']['historical_prices_file']
    if not os.path.exists(price_file):
        logger.warning(f"Historical prices not found at {price_file}, generating sample data")
        price_df = generate_sample_historical_data()
        price_df.to_csv(price_file)
    else:
        price_df = pd.read_csv(price_file, parse_dates=[0], index_col=0)
    
    # Estimate market parameters
    estimator = MarketDataEstimator(risk_free_rate=config['market_data']['risk_free_rate'])
    market_params = estimator.estimate_from_prices(price_df)
    
    # Run Monte Carlo simulation
    mc_engine = MonteCarloEngine(sim_config, market_params)
    paths = mc_engine.simulate_paths()
    
    # Compute simulation diagnostics
    diagnostics = mc_engine.compute_simulation_diagnostics(paths)
    
    # Calculate portfolio values
    risk_free_rate = config['market_data']['risk_free_rate']
    exposure_calc = ExposureCalculator(positions, csa_terms, risk_free_rate)
    portfolio_values = exposure_calc.compute_portfolio_values(paths, market_params.spot_prices)
    
    # Compute time grid
    time_grid = np.linspace(0, sim_config.horizon_days / 252, sim_config.time_steps + 1)
    
    # Compute exposure profile
    pfe_percentiles = config['exposure_metrics']['pfe_percentiles']
    exposure_profile = exposure_calc.compute_exposure_profile(
        portfolio_values, 
        time_grid, 
        pfe_percentiles,
        sim_config.confidence_level
    )
    
    # Apply collateral if CSA exists
    if csa_terms and csa_terms.has_csa:
        collateralized_exposures = exposure_calc.apply_collateral(
            np.maximum(portfolio_values, 0), time_grid
        )
        exposure_profile.collateralized_ee = collateralized_exposures.mean(axis=0)
        exposure_profile.collateralized_pfe = {
            p: np.percentile(collateralized_exposures, p, axis=0) for p in pfe_percentiles
        }
    
    # Calculate CVA
    cva_result = None
    if config['exposure_metrics']['calculate_cva']:
        cva_calc = CVACalculator(counterparty, risk_free_rate)
        cva_result = cva_calc.compute_cva(exposure_profile)
    
    # Analyze Wrong-Way Risk
    wwr_metrics = None
    if config['exposure_metrics']['calculate_wrong_way_risk']:
        wwr_analyzer = WrongWayRiskAnalyzer(counterparty)
        wwr_metrics = wwr_analyzer.analyze_wwr(portfolio_values)
    
    # Generate reports
    output_config = config['output']
    reporter = ReportGenerator(
        output_dir=output_config['directory'],
        plot_format=output_config['plot_format'],
        plot_dpi=output_config['plot_dpi']
    )
    
    reporter.save_exposure_profile(exposure_profile, output_config['exposure_profile_csv'])
    reporter.save_simulation_diagnostics(diagnostics, output_config['simulation_diagnostics_csv'])
    
    if cva_result:
        reporter.save_cva_summary(cva_result, counterparty, output_config['cva_summary_csv'])
    
    if output_config['generate_plots']:
        reporter.plot_exposure_profile(exposure_profile, cva_result, wwr_metrics)
    
    reporter.generate_summary_json(exposure_profile, cva_result, wwr_metrics, market_params, sim_config)
    
    logger.info("=" * 60)
    logger.info("EXPOSURE MODEL EXECUTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Peak Exposure: ${exposure_profile.peak_exposure:,.0f}")
    logger.info(f"Effective EPE: ${exposure_profile.effective_epe:,.0f}")
    if cva_result:
        logger.info(f"CVA: ${cva_result.cva_value:,.0f}")
    logger.info(f"Outputs saved to: {output_config['directory']}")


if __name__ == "__main__":
    main()