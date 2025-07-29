"""
Basic test for transient time functionality without requiring MATLAB data.

This test creates synthetic stable systems and verifies that get_transient_time
works correctly, providing a fallback when MATLAB reference data is not available.
"""

import pytest
import numpy as np
import torch

from crnn.models.base import Linear
from crnn.systemtheory.analysis import get_transient_time


class TestTransientTimeBasic:
    """Basic tests for transient time calculation without MATLAB dependencies."""
    
    def test_simple_stable_system(self):
        """Test transient time calculation for a simple stable system."""
        # Create a simple 2x2 stable system
        A = torch.tensor([[0.5, 0.1], [0.0, 0.8]], dtype=torch.float64)
        B = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
        C = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        D = torch.tensor([[0.0]], dtype=torch.float64)
        dt = torch.tensor(0.1)
        
        linear_sys = Linear(A, B, C, D, dt)
        
        # Verify system is stable
        eigenvalues = torch.linalg.eigvals(A)
        max_eig = torch.max(torch.abs(eigenvalues)).item()
        assert max_eig < 1.0, f"System should be stable, but max|λ| = {max_eig}"
        
        # Calculate transient time
        transient_time = get_transient_time(linear_sys, n_max=1000)
        
        # Basic sanity checks
        assert transient_time > 0, "Transient time should be positive"
        assert transient_time < 100, "Transient time should be reasonable (< 100 time units)"
        
        print(f"✓ Simple system transient time: {transient_time}")
    
    def test_very_stable_system(self):
        """Test transient time for a very stable (fast) system."""
        # Create a very stable system (eigenvalues close to 0)
        A = torch.tensor([[0.1, 0.0], [0.0, 0.2]], dtype=torch.float64)
        B = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
        C = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        D = torch.tensor([[0.0]], dtype=torch.float64)
        dt = torch.tensor(0.1)
        
        linear_sys = Linear(A, B, C, D, dt)
        
        # This system should have a short transient time
        transient_time = get_transient_time(linear_sys, n_max=1000)
        
        assert transient_time > 0
        assert transient_time < 50, f"Very stable system should have short transient time, got {transient_time}"
        
        print(f"✓ Very stable system transient time: {transient_time}")
    
    def test_slower_stable_system(self):
        """Test transient time for a slower (less stable) system."""
        # Create a system with eigenvalues closer to 1 (slower decay)
        A = torch.tensor([[0.9, 0.0], [0.0, 0.95]], dtype=torch.float64)
        B = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
        C = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        D = torch.tensor([[0.0]], dtype=torch.float64)
        dt = torch.tensor(0.1)
        
        linear_sys = Linear(A, B, C, D, dt)
        
        # This system should have a longer transient time
        transient_time = get_transient_time(linear_sys, n_max=2000)
        
        assert transient_time > 0
        # Slower systems should have longer transient times
        print(f"✓ Slower system transient time: {transient_time}")
    
    def test_system_dimensions(self):
        """Test that get_transient_time works with different system dimensions."""
        # Test with different input/output dimensions
        test_cases = [
            # (nx, nu, ny)
            (1, 1, 1),  # SISO
            (2, 1, 1),  # 2nd order SISO 
            (3, 2, 1),  # MISO
            (2, 1, 2),  # SIMO
            (3, 2, 2),  # MIMO
        ]
        
        for nx, nu, ny in test_cases:
            # Create random stable system
            A = torch.randn(nx, nx, dtype=torch.float64) * 0.3
            # Make sure it's stable by scaling eigenvalues
            eigenvalues = torch.linalg.eigvals(A)
            max_eig = torch.max(torch.abs(eigenvalues)).item()
            if max_eig >= 1.0:
                A = A * (0.8 / max_eig)  # Scale to be stable
            
            B = torch.randn(nx, nu, dtype=torch.float64)
            C = torch.randn(ny, nx, dtype=torch.float64)
            D = torch.zeros(ny, nu, dtype=torch.float64)
            dt = torch.tensor(0.1)
            
            linear_sys = Linear(A, B, C, D, dt)
            print(nu, ny, nx)
            
            # Calculate transient time
            transient_time = get_transient_time(linear_sys, n_max=1000)
            
            assert (transient_time > 0).all(), f"Transient time should be positive for system ({nx},{nu},{ny})"
            print(f"✓ System ({nx},{nu},{ny}) transient time: {transient_time}")
    
    def test_integrator_system_handling(self):
        """Test handling of marginally stable systems (integrators)."""
        # Create a system with an integrator (eigenvalue = 1)
        A = torch.tensor([[1.0, 1.0], [0.0, 0.8]], dtype=torch.float64)
        B = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        C = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        D = torch.tensor([[0.0]], dtype=torch.float64)
        
        linear_sys = Linear(A, B, C, D)
        
        # This should either work or raise a reasonable error
        try:
            transient_time = get_transient_time(linear_sys, n_max=1000)
            # If it works, transient time should be large or the max allowed
            print(f"✓ Integrator system transient time: {transient_time}")
        except ValueError as e:
            # It's acceptable to fail on marginally stable systems
            print(f"✓ Integrator system appropriately rejected: {e}")
            assert "system is not stable" in str(e)


class TestTransientTimeEdgeCases:
    """Test edge cases and error handling for transient time calculation."""
    
    def test_unstable_system(self):
        """Test that unstable systems are handled appropriately."""
        # Create an unstable system (eigenvalue > 1)
        A = torch.tensor([[1.1, 0.0], [0.0, 0.5]], dtype=torch.float64)
        B = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
        C = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        D = torch.tensor([[0.0]], dtype=torch.float64)
        dt = torch.tensor(0.1)
        
        linear_sys = Linear(A, B, C, D, dt)
        
        # This should either work (with large transient time) or raise an error
        try:
            transient_time = get_transient_time(linear_sys, n_max=1000)
            print(f"Unstable system transient time: {transient_time}")
            # For unstable systems, we might get the maximum time
        except ValueError as e:
            print(f"✓ Unstable system appropriately handled: {e}")
    
    def test_very_small_n_max(self):
        """Test behavior with very small n_max parameter."""
        # Simple stable system
        A = torch.tensor([[0.5]], dtype=torch.float64)
        B = torch.tensor([[1.0]], dtype=torch.float64)
        C = torch.tensor([[1.0]], dtype=torch.float64)
        D = torch.tensor([[0.0]], dtype=torch.float64)
        
        linear_sys = Linear(A, B, C, D)
        
        # Test with very small n_max
        try:
            transient_time = get_transient_time(linear_sys, n_max=50)
            print(f"Small n_max transient time: {transient_time}")
        except ValueError as e:
            print(f"✓ Small n_max appropriately handled: {e}")


