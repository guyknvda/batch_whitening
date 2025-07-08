#!/usr/bin/env python3
"""
Script to extract trial data from pickled Optuna studies with version compatibility issues.
"""

import pickle
import os
import optuna
from optuna.trial import TrialState

def safe_extract_trials(pickle_file):
    """
    Try to extract trial data from a pickled study, handling version incompatibilities.
    """
    print(f"Attempting to extract data from {pickle_file}...")
    
    try:
        # Method 1: Try direct loading
        with open(pickle_file, 'rb') as f:
            study = pickle.load(f)
        print(f"✓ Successfully loaded {pickle_file} directly")
        return study
    except Exception as e:
        print(f"✗ Direct loading failed: {e}")
    
    try:
        # Method 2: Try loading with custom unpickler
        import sys
        import types
        
        class CompatibilityUnpickler(pickle.Unpickler):
            def load_build(self):
                # Override problematic class construction
                try:
                    return super().load_build()
                except TypeError as e:
                    if "_ParzenEstimatorParameters" in str(e):
                        # Skip this object and return None
                        return None
                    raise
        
        with open(pickle_file, 'rb') as f:
            unpickler = CompatibilityUnpickler(f)
            study = unpickler.load()
        
        print(f"✓ Successfully loaded {pickle_file} with compatibility unpickler")
        return study
        
    except Exception as e:
        print(f"✗ Compatibility unpickler failed: {e}")
    
    # Method 3: Manual trial extraction (if we can access the raw data)
    try:
        with open(pickle_file, 'rb') as f:
            data = f.read()
        
        # Try to find trial data patterns in the binary data
        print(f"File size: {len(data)} bytes")
        print("Could not extract trials automatically. Manual inspection may be needed.")
        return None
        
    except Exception as e:
        print(f"✗ All methods failed: {e}")
        return None

def create_new_study_from_trials(trials, study_name="recovered_study"):
    """
    Create a new Optuna study from extracted trial data.
    """
    if not trials:
        return None
    
    # Create a new study
    study = optuna.create_study(direction='minimize', study_name=study_name)
    
    # Add trials manually
    for trial in trials:
        try:
            # Create a new trial with the same parameters and value
            study.enqueue_trial(trial.params)
            new_trial = study.ask()
            study.tell(new_trial, trial.value if hasattr(trial, 'value') else trial.values[0])
        except Exception as e:
            print(f"Warning: Could not add trial {trial.number}: {e}")
    
    return study

def main():
    pickle_files = ['study_bn.pkl', 'study_x.pkl', 'study_xx.pkl', 'study.pkl']
    
    for pickle_file in pickle_files:
        if os.path.exists(pickle_file):
            print(f"\n{'='*50}")
            print(f"Processing {pickle_file}")
            print(f"{'='*50}")
            
            study = safe_extract_trials(pickle_file)
            
            if study and hasattr(study, 'trials'):
                print(f"✓ Extracted {len(study.trials)} trials")
                print(f"✓ Best trial: {study.best_trial.value if study.best_trial else 'None'}")
                print(f"✓ Parameters: {list(study.trials[0].params.keys()) if study.trials else 'None'}")
                
                # Save as a new compatible study
                new_filename = f"compatible_{pickle_file}"
                try:
                    with open(new_filename, 'wb') as f:
                        pickle.dump(study, f)
                    print(f"✓ Saved compatible version as {new_filename}")
                except Exception as e:
                    print(f"✗ Could not save compatible version: {e}")
            else:
                print("✗ Could not extract study data")

if __name__ == "__main__":
    main() 