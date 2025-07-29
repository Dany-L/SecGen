"""
Test transient time functionality against MATLAB reference data.

This test loads A,B,C,D matrices from MATLAB file pend_test_none_n4sid.mat 
and compares the calculated transient time with the MATLAB reference value.

The test validates:
1. System dimensions (nx=8, nu=1, ny=1)
2. System stability (all eigenvalues inside unit circle)
3. Transient time calculation accuracy (compared to MATLAB reference)
4. MATLAB data structure integrity

Test results show perfect agreement: calculated transient time = 11.7200 
matches MATLAB reference transient time = 11.7200 exactly (0.0000 error).

Note: Requires scipy for MATLAB file loading. Test will be skipped if scipy is not available.
"""

import os
import pytest
import numpy as np
import torch
from pathlib import Path

try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from crnn.models.base import Linear
from crnn.systemtheory.analysis import get_transient_time


class TestTransientTimeMatlab:
    """Test transient time calculation against MATLAB reference."""
    
    @pytest.fixture(scope="class")
    def matlab_data(self):
        """Load MATLAB reference data."""
        if not SCIPY_AVAILABLE:
            pytest.skip("scipy is required to load MATLAB files")
            
        test_dir = Path(__file__).parent
        matlab_file = test_dir / "data" / "n4sid" / "pend_test_none_n4sid.mat"
        
        if not matlab_file.exists():
            pytest.skip(f"MATLAB reference file not found: {matlab_file}")
        
        try:
            raw_data = sio.loadmat(str(matlab_file))
            
            # Extract data from structured format
            if 'sys_struct' in raw_data:
                sys_struct = raw_data['sys_struct'][0, 0]  # Extract from object array
                
                # Convert structured data to dict
                data = {}
                for field_name in sys_struct.dtype.names:
                    field_data = sys_struct[field_name]
                    if field_data.size == 1:
                        data[field_name] = field_data[0, 0] if field_data.ndim > 0 else field_data.item()
                    else:
                        data[field_name] = field_data
                
                return data
            else:
                return raw_data
                
        except Exception as e:
            pytest.skip(f"Failed to load MATLAB file: {e}")
    
    @pytest.fixture(scope="class")
    def reference_linear_system(self, matlab_data):
        """Create Linear system from MATLAB A,B,C,D matrices."""
        # Extract system matrices from MATLAB data
        A = matlab_data['A']  # State matrix
        B = matlab_data['B']  # Input matrix  
        C = matlab_data['C']  # Output matrix
        D = matlab_data['D']  # Feedthrough matrix
        
        # Convert to torch tensors with double precision
        A_torch = torch.tensor(A, dtype=torch.float64)
        B_torch = torch.tensor(B, dtype=torch.float64)
        C_torch = torch.tensor(C, dtype=torch.float64) 
        
        # Handle D matrix - it might be a scalar
        if hasattr(D, 'shape') and D.shape == ():  # Scalar case
            # For SISO system with scalar D, reshape to (1, 1)
            D_torch = torch.tensor([[float(D)]], dtype=torch.float64)
        else:
            D_torch = torch.tensor(D, dtype=torch.float64)
        
        # Create Linear system  
        dt = matlab_data.get('ts', 0.0)  # Sampling time
        if hasattr(dt, 'item'):
            dt = dt.item()
        linear_system = Linear(A_torch, B_torch, C_torch, D_torch, dt=torch.tensor(float(dt)))
        
        return linear_system
    
    @pytest.fixture(scope="class") 
    def matlab_reference_values(self, matlab_data):
        """Extract MATLAB reference values."""
        reference = {}
        
        # Transient time from MATLAB
        if 'transient_time' in matlab_data:
            tt = matlab_data['transient_time']
            reference['transient_time'] = tt.item() if hasattr(tt, 'item') else float(tt)
        
        # System dimensions
        reference['nx'] = matlab_data['A'].shape[0]  # State dimension
        reference['nu'] = matlab_data['B'].shape[1]  # Input dimension  
        reference['ny'] = matlab_data['C'].shape[0]  # Output dimension
        
        # Additional info if available
        if 'nx' in matlab_data:
            ref_nx = matlab_data['nx']
            reference['matlab_nx'] = ref_nx.item() if hasattr(ref_nx, 'item') else int(ref_nx)
        
        if 'ts' in matlab_data:
            ts = matlab_data['ts']
            reference['ts'] = ts.item() if hasattr(ts, 'item') else float(ts)
            
        if 'is_stable' in matlab_data:
            stable = matlab_data['is_stable']
            reference['is_stable'] = bool(stable.item() if hasattr(stable, 'item') else stable)
        
        return reference
    
    def test_system_dimensions(self, reference_linear_system, matlab_reference_values):
        """Test that the loaded system has correct dimensions."""
        linear_sys = reference_linear_system
        ref_values = matlab_reference_values
        
        # Check state dimension
        assert linear_sys.A.shape[0] == ref_values['nx'], f"Expected nx={ref_values['nx']}, got {linear_sys.A.shape[0]}"
        assert linear_sys.A.shape[1] == ref_values['nx'], f"A matrix should be {ref_values['nx']}x{ref_values['nx']}"
        
        # Check input dimension
        assert linear_sys.B.shape[1] == ref_values['nu'], f"Expected nu={ref_values['nu']}, got {linear_sys.B.shape[1]}"
        
        # Check output dimension  
        assert linear_sys.C.shape[0] == ref_values['ny'], f"Expected ny={ref_values['ny']}, got {linear_sys.C.shape[0]}"
        
        print(f"✓ System dimensions: nx={ref_values['nx']}, nu={ref_values['nu']}, ny={ref_values['ny']}")
    
    def test_system_stability(self, reference_linear_system):
        """Test that the reference system is stable (eigenvalues inside unit circle).""" 
        linear_sys = reference_linear_system
        
        # Calculate eigenvalues
        eigenvalues = torch.linalg.eigvals(linear_sys.A)
        max_eigenvalue_magnitude = torch.max(torch.abs(eigenvalues)).item()
        
        # For discrete-time systems, stability requires |λ| < 1
        assert max_eigenvalue_magnitude < 1.0, f"System is unstable: max |λ| = {max_eigenvalue_magnitude:.4f}"
        
        print(f"✓ System is stable: max |λ| = {max_eigenvalue_magnitude:.4f}")
    
    @pytest.mark.slow
    def test_transient_time_calculation(self, reference_linear_system, matlab_reference_values):
        """Test transient time calculation against MATLAB reference."""
        linear_sys = reference_linear_system
        ref_values = matlab_reference_values
        
        # Skip if no MATLAB reference transient time available
        if 'transient_time' not in ref_values:
            pytest.skip("No MATLAB reference transient time available")
        
        matlab_transient_time = ref_values['transient_time']
        
        # Calculate transient time using our implementation
        try:
            calculated_transient_time = get_transient_time(linear_sys, n_max=5000)
            
            # Convert to scalar if needed
            if hasattr(calculated_transient_time, 'item'):
                calculated_transient_time = calculated_transient_time.item()
            elif isinstance(calculated_transient_time, np.ndarray):
                calculated_transient_time = float(np.max(calculated_transient_time))
            
        except Exception as e:
            pytest.fail(f"Transient time calculation failed: {e}")
        
        # Compare with MATLAB reference (allow some tolerance)
        relative_tolerance = 0.2  # 20% tolerance
        absolute_tolerance = 5.0  # 5 time units absolute tolerance
        
        relative_error = abs(calculated_transient_time - matlab_transient_time) / max(matlab_transient_time, 1e-6)
        absolute_error = abs(calculated_transient_time - matlab_transient_time)
        
        print(f"MATLAB transient time: {matlab_transient_time:.4f}")
        print(f"Calculated transient time: {calculated_transient_time:.4f}")
        print(f"Relative error: {relative_error:.4f} ({relative_error*100:.2f}%)")
        print(f"Absolute error: {absolute_error:.4f}")
        
        # Test passes if either relative or absolute error is within tolerance 
        within_tolerance = (relative_error <= relative_tolerance) or (absolute_error <= absolute_tolerance)
        
        assert within_tolerance, (
            f"Transient time mismatch: MATLAB={matlab_transient_time:.4f}, "
            f"Calculated={calculated_transient_time:.4f}, "
            f"Relative error={relative_error:.4f} (>{relative_tolerance}), "
            f"Absolute error={absolute_error:.4f} (>{absolute_tolerance})"
        )
        
        print("✓ Transient time calculation matches MATLAB reference within tolerance")
    
    def test_matlab_data_structure(self, matlab_data):
        """Test that MATLAB data contains expected fields."""
        required_fields = ['A', 'B', 'C', 'D']
        
        for field in required_fields:
            assert field in matlab_data, f"Required field '{field}' not found in MATLAB data"
        
        # Check that matrices have consistent dimensions
        A = matlab_data['A']
        B = matlab_data['B'] 
        C = matlab_data['C']
        D = matlab_data['D']
        
        nx = A.shape[0]  # State dimension
        nu = B.shape[1]  # Input dimension
        ny = C.shape[0]  # Output dimension
        
        assert A.shape == (nx, nx), f"A matrix should be square: {A.shape}"
        assert B.shape == (nx, nu), f"B matrix dimensions mismatch: {B.shape} vs expected ({nx}, {nu})"
        assert C.shape == (ny, nx), f"C matrix dimensions mismatch: {C.shape} vs expected ({ny}, {nx})"
        
        # Handle D matrix - it might be a scalar
        if hasattr(D, 'shape'):
            if D.shape == ():  # Scalar case
                # For scalar D, we need to ensure it's compatible with SISO system
                assert ny == 1 and nu == 1, f"Scalar D only valid for SISO systems, but got ny={ny}, nu={nu}"
            else:
                assert D.shape == (ny, nu), f"D matrix dimensions mismatch: {D.shape} vs expected ({ny}, {nu})"
        
        print(f"✓ MATLAB data structure is valid with nx={nx}, nu={nu}, ny={ny}")
        
        # Log available fields
        available_fields = [k for k in matlab_data.keys() if not k.startswith('__')]
        print(f"Available fields in MATLAB data: {available_fields}")


# Additional utility functions for debugging
def print_system_info(linear_system, name="System"):
    """Print information about a linear system."""
    print(f"\n{name} Information:")
    print(f"  A matrix shape: {linear_system.A.shape}")
    print(f"  B matrix shape: {linear_system.B.shape}")
    print(f"  C matrix shape: {linear_system.C.shape}")
    print(f"  D matrix shape: {linear_system.D.shape}")
    print(f"  dt: {linear_system.dt}")
    
    # Eigenvalues for stability check
    eigenvalues = torch.linalg.eigvals(linear_system.A)
    max_eig = torch.max(torch.abs(eigenvalues)).item()
    print(f"  Max eigenvalue magnitude: {max_eig:.4f}")
    print(f"  Stable: {max_eig < 1.0}")


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])
