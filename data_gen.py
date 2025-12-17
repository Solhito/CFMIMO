import numpy as np
import torch
import pickle
import os
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ChannelDataGenerator:
    """Channel data generator"""
    
    def __init__(self, config: Dict):
        """
        Initialize data generator
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        
        # System parameters
        self.L = config["L"]  # Number of APs
        self.K = config["K"]  # Number of UEs
        self.N = config["N"]  # Number of antennas per AP
        self.tau_p = config["tau_p"]  # Pilot length
        self.tau_c = config["tau_c"]  # Coherence block length
        self.tau_d = config["tau_d"]  # Downlink data length
        
        # Network topology parameters
        self.area_len = config.get("area_len", 1000.0)  # Area size (m)
        self.alpha = config.get("alpha", 3.5)  # Path loss exponent
        self.shadow_std_db = config.get("shadow_std_db", 8.0)  # Shadow fading standard deviation (dB)
        self.rho_corr = config.get("rho_corr", 0.3)  # Antenna correlation coefficient
        
        # Power and capacity parameters
        self.sigma2 = config.get("sigma2", 1e-2)  # Noise variance
        self.P_max = config.get("P_max", 1.0)  # Maximum transmit power per AP
        self.C_max = config.get("C_max", 2.0)  # Fronthaul capacity constraint
        self.eta_pilot = config.get("eta_pilot", 0.1)  # Uplink pilot power
        
        # Dataset sizes
        self.train_size = config.get("train_size", 500)
        self.val_size = config.get("val_size", 100)
        self.test_size = config.get("test_size", 100)
        
        # Data saving path
        self.data_dir = config.get("data_dir", "./data")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def exp_corr_matrix(self, N: int, rho: float) -> np.ndarray:
        """
        Generate exponential correlation matrix
        
        Args:
            N: Number of antennas
            rho: Correlation coefficient
            
        Returns:
            R: (N, N) correlation matrix
        """
        if N == 1:
            return np.array([[1.0]], dtype=np.float32)
        
        idx = np.arange(N)
        R = rho ** np.abs(np.subtract.outer(idx, idx))
        return R.astype(np.float32)
    
    def generate_layout(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random layout of APs and UEs
        
        Returns:
            ap_pos: (L, 2) AP positions
            ue_pos: (K, 2) UE positions
        """
        # APs uniformly distributed in the area
        ap_pos = np.random.uniform(0.0, self.area_len, size=(self.L, 2))
        
        # UEs randomly distributed with some hotspots
        ue_pos = np.zeros((self.K, 2))
        for k in range(self.K):
            # 80% UEs uniformly distributed, 20% in hotspots
            if np.random.rand() < 0.8:
                ue_pos[k] = np.random.uniform(0.0, self.area_len, size=2)
            else:
                # Randomly select a hotspot center
                hotspot_center = np.random.uniform(0.2*self.area_len, 0.8*self.area_len, size=2)
                ue_pos[k] = hotspot_center + np.random.normal(0, 0.1*self.area_len, size=2)
                ue_pos[k] = np.clip(ue_pos[k], 0, self.area_len)
        
        return ap_pos.astype(np.float32), ue_pos.astype(np.float32)
    
    def compute_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute Euclidean distance between two points"""
        return np.linalg.norm(pos1 - pos2)
    
    def compute_beta_kl(self, ap_pos: np.ndarray, ue_pos: np.ndarray) -> np.ndarray:
        """
        Compute large-scale fading coefficients β_{kl}
        Follow 3GPP Urban Microcell model
        
        Returns:
            betas: (L, K) Large-scale fading coefficients (linear scale)
        """
        L, K = ap_pos.shape[0], ue_pos.shape[0]
        betas = np.zeros((L, K), dtype=np.float32)
        
        # Reference distance and path loss parameters
        d0 = 10.0  # Reference distance (m)
        pl_d0_db = 30.0  # Path loss at reference distance (dB)
        
        for l in range(L):
            for k in range(K):
                # Compute distance
                d = max(self.compute_distance(ap_pos[l], ue_pos[k]), d0)
                
                # Path loss (dB)
                pl_db = pl_d0_db + 10.0 * self.alpha * np.log10(d / d0)
                
                # Shadow fading (dB)
                shadow_db = np.random.normal(0.0, self.shadow_std_db)
                
                # Total loss (dB)
                total_loss_db = pl_db + shadow_db
                
                # Convert to linear scale
                betas[l, k] = 10 ** (-total_loss_db / 10.0)
        
        return betas
    
    def generate_R_matrices(self, betas: np.ndarray) -> np.ndarray:
        """
        Generate spatial correlation matrices R_{kl}
        R_{kl} = β_{kl} * R_base, where R_base is exponential correlation matrix
        
        Args:
            betas: (L, K) Large-scale fading coefficients
            
        Returns:
            R_all: (L, K, N, N) Spatial correlation matrices (complex)
        """
        L, K = betas.shape
        
        # Generate base correlation matrix (Hermitian)
        R_base = self.exp_corr_matrix(self.N, self.rho_corr)
        
        # Make R_base complex and Hermitian
        R_base_complex = R_base.astype(np.complex64)
        # Ensure it's Hermitian (symmetric for real matrix)
        R_base_complex = (R_base_complex + R_base_complex.conj().T) / 2
        
        # Scale by β_{kl} to get R_{kl}
        R_all = np.zeros((L, K, self.N, self.N), dtype=np.complex64)
        
        for l in range(L):
            for k in range(K):
                beta = betas[l, k]
                R_all[l, k] = beta * R_base_complex
        
        return R_all
    
    def generate_scenario(self) -> Dict:
        """
        Generate a complete scenario data
        
        Returns:
            Dictionary containing scenario data
        """
        # Generate layout
        ap_pos, ue_pos = self.generate_layout()
        
        # Compute large-scale fading
        betas = self.compute_beta_kl(ap_pos, ue_pos)
        
        # Generate spatial correlation matrices
        R_all = self.generate_R_matrices(betas)
        
        # Generate uplink pilot power vector η (size K)
        eta = np.ones(self.K, dtype=np.float32) * self.eta_pilot
        
        # System configuration
        system_config = {
            "R_all": R_all.astype(np.complex64),
            "betas": betas.astype(np.float32),
            "ap_pos": ap_pos,
            "ue_pos": ue_pos,
            "eta": eta.astype(np.float32),  # Uplink pilot power
            "sigma2": self.sigma2,
            "P_max": self.P_max,
            "C_max": self.C_max,
            "L": self.L,
            "K": self.K,
            "N": self.N,
            "tau_p": self.tau_p,
            "tau_c": self.tau_c,
            "tau_d": self.tau_d
        }
        
        return system_config
    
    def generate_dataset(self, size: int, dataset_type: str = "train") -> List[Dict]:
        """
        Generate dataset of specified size
        
        Args:
            size: Dataset size
            dataset_type: Dataset type
            
        Returns:
            dataset: List of data samples
        """
        print(f"Generating {dataset_type} dataset, size: {size}")
        
        dataset = []
        for i in range(size):
            if i % 100 == 0:
                print(f"  Generating sample {i}/{size}...")
            
            scenario = self.generate_scenario()
            dataset.append(scenario)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str):
        """Save dataset to file"""
        filepath = os.path.join(self.data_dir, filename)
        
        saved_data = {
            "config": self.config,
            "data": dataset
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(saved_data, f)
        
        print(f"Dataset saved to: {filepath}")
    
    def create_datasets(self):
        """Create all datasets"""
        print("=" * 60)
        print("Starting Cell-Free Massive MIMO dataset generation")
        print("=" * 60)
        
        # Generate training set
        train_data = self.generate_dataset(self.train_size, "train")
        self.save_dataset(train_data, "train_dataset.pkl")
        
        # Generate validation set
        val_data = self.generate_dataset(self.val_size, "validation")
        self.save_dataset(val_data, "val_dataset.pkl")
        
        # Generate test set
        test_data = self.generate_dataset(self.test_size, "test")
        self.save_dataset(test_data, "test_dataset.pkl")
        
        # Save configuration
        config_file = os.path.join(self.data_dir, "system_config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        print("\nDataset generation completed!")

def generate_default_datasets():
    """Generate datasets with default configuration"""
    config = {
        # System parameters
        "L": 8,  # Number of APs
        "K": 4,  # Number of UEs
        "N": 2,  # Number of antennas per AP
        "tau_p": 3,
        "tau_c": 200,
        "tau_d": 197,
        
        # Network topology parameters
        "area_len": 1000.0,
        "alpha": 3.5,
        "shadow_std_db": 8.0,
        "rho_corr": 0.3,
        
        # Power and capacity parameters
        "sigma2": 1e-15,
        "P_max": 1.0,
        "C_max": 2.0,
        "eta_pilot": 0.1,  # Uplink pilot power
        
        # Dataset sizes
        "train_size": 500,
        "val_size": 100,
        "test_size": 100,
        
        # Data saving path
        "data_dir": "./data/cellfree_mimo"
    }
    
    generator = ChannelDataGenerator(config)
    generator.create_datasets()

if __name__ == "__main__":
    generate_default_datasets()