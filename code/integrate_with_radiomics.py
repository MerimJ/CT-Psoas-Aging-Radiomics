#!/usr/bin/env python3
"""
Integrate Predicted Masks with Radiomics (Phase B)
Updates mask_path in pairs_df to use predicted .nii files, then reruns Phase B.
Compares auto vs manual features to assess automation impact.
"""

import os
if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
from pathlib import Path
import gc

# Import Phase B components (simplified version)
try:
    import SimpleITK as sitk
    from radiomics import featureextractor
except ImportError:
    print("Warning: SimpleITK or PyRadiomics not installed.")
    print("Install with: pip install SimpleITK pyradiomics")
    sitk = None
    featureextractor = None


def compute_custom_features_lowmem(ct_img_sitk, mask_img_sitk, voxel_spacing):
    """Compute custom features (from Phase B)."""
    ct_data = sitk.GetArrayFromImage(ct_img_sitk)
    mask_data = sitk.GetArrayFromImage(mask_img_sitk)
    masked_img = ct_data[mask_data > 0]
    
    del ct_data, mask_data
    gc.collect()
    
    if len(masked_img) == 0:
        return {
            'volume_mm3': 0, 'hu_mean': 0, 'pct_hu_lt_0': 0,
            'pct_hu_0_29': 0, 'pct_hu_30_100': 0, 'muscle_quality_index': 0
        }
    
    hu_mean = np.mean(masked_img)
    volume_mm3 = len(masked_img) * np.prod(voxel_spacing)
    total_voxels = len(masked_img)
    
    pct_lt_0 = np.sum(masked_img < 0) / total_voxels * 100
    pct_0_29 = np.sum((masked_img >= 0) & (masked_img <= 29)) / total_voxels * 100
    pct_30_100 = np.sum((masked_img >= 30) & (masked_img <= 100)) / total_voxels * 100
    mqi = hu_mean
    
    del masked_img
    gc.collect()
    
    return {
        'volume_mm3': volume_mm3,
        'hu_mean': hu_mean,
        'pct_hu_lt_0': pct_lt_0,
        'pct_hu_0_29': pct_0_29,
        'pct_hu_30_100': pct_30_100,
        'muscle_quality_index': mqi
    }


def extract_features_patient(ct_path, mask_path, extractor=None):
    """Extract features for a single patient."""
    if sitk is None:
        raise ImportError("SimpleITK not available")
    
    ct_img_sitk = sitk.ReadImage(ct_path)
    mask_img_sitk = sitk.ReadImage(mask_path)
    voxel_spacing = ct_img_sitk.GetSpacing()
    
    # Custom features
    custom_features = compute_custom_features_lowmem(ct_img_sitk, mask_img_sitk, voxel_spacing)
    
    # Radiomics features (if extractor available)
    glcm_features = {}
    if extractor is not None:
        try:
            radiomics_features = extractor.execute(ct_img_sitk, mask_img_sitk)
            glcm_features = {k: v for k, v in radiomics_features.items() if 'glcm' in k.lower()}
        except Exception as e:
            print(f"  Warning: Radiomics extraction failed: {e}")
    
    del ct_img_sitk, mask_img_sitk
    gc.collect()
    
    return {**custom_features, **glcm_features}


def compare_features(manual_features, auto_features, patient_id):
    """Compare manual vs automated features."""
    comparison = {
        'patient_id': patient_id
    }
    
    # Key features to compare
    key_features = ['volume_mm3', 'hu_mean', 'pct_hu_lt_0', 'pct_hu_0_29', 
                   'pct_hu_30_100', 'muscle_quality_index']
    
    for feat in key_features:
        if feat in manual_features and feat in auto_features:
            manual_val = manual_features[feat]
            auto_val = auto_features[feat]
            
            if manual_val != 0:
                diff_pct = abs(auto_val - manual_val) / abs(manual_val) * 100
            else:
                diff_pct = 100 if auto_val != 0 else 0
            
            comparison[f'{feat}_manual'] = manual_val
            comparison[f'{feat}_auto'] = auto_val
            comparison[f'{feat}_diff_pct'] = diff_pct
    
    return comparison


def main():
    base_path = Path(__file__).parent
    results_path = base_path / 'Results'
    pairs_csv = results_path / 'pairs.csv'
    
    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs.csv not found. Run Phase A first.")
    
    # Load pairs
    pairs_df = pd.read_csv(pairs_csv)
    
    # Fix paths
    def fix_path(path_str):
        if pd.isna(path_str) or not path_str:
            return None
        path_str = str(path_str)
        if os.path.exists(path_str):
            return path_str
        if '/content/drive/MyDrive/Psoas project/' in path_str:
            relative = path_str.replace('/content/drive/MyDrive/Psoas project/', '')
            local = base_path / relative
            if local.exists():
                return str(local)
        return path_str
    
    pairs_df['ct_path'] = pairs_df['ct_path'].apply(fix_path)
    pairs_df['mask_path'] = pairs_df['mask_path'].apply(fix_path)
    
    # Get test patients (last 20%)
    split_idx = int(len(pairs_df) * 0.8)
    test_df = pairs_df.iloc[split_idx:].reset_index(drop=True)
    
    print("="*60)
    print("Integrating Predicted Masks with Radiomics")
    print("="*60)
    print(f"Test patients: {len(test_df)}")
    
    # Initialize radiomics extractor (if available)
    extractor = None
    if featureextractor is not None:
        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=25)
            extractor.disableAllFeatures()
            extractor.enableFeatureClassByName('glcm')
            extractor.enableImageTypeByName('Original')
            print("Radiomics extractor initialized")
        except Exception as e:
            print(f"Warning: Could not initialize radiomics extractor: {e}")
    
    # Process test patients
    comparisons = []
    
    for idx, row in test_df.iterrows():
        patient_id = row['patient_id']
        ct_path = row['ct_path']
        manual_mask_path = row['mask_path']
        pred_mask_path = results_path / f'predicted_mask_patient{patient_id}.nii'
        
        if not os.path.exists(ct_path):
            print(f"Skipping patient {patient_id}: CT not found")
            continue
        
        if not os.path.exists(pred_mask_path):
            print(f"Skipping patient {patient_id}: Prediction not found. Run test_model.py first.")
            continue
        
        print(f"\nProcessing patient {patient_id}...")
        
        try:
            # Extract features from manual mask
            if os.path.exists(manual_mask_path):
                print(f"  Extracting features from manual mask...")
                manual_features = extract_features_patient(ct_path, manual_mask_path, extractor)
            else:
                print(f"  Warning: Manual mask not found, skipping comparison")
                continue
            
            # Extract features from predicted mask
            print(f"  Extracting features from predicted mask...")
            auto_features = extract_features_patient(ct_path, str(pred_mask_path), extractor)
            
            # Compare
            comparison = compare_features(manual_features, auto_features, patient_id)
            comparisons.append(comparison)
            
            # Print key differences
            if 'volume_mm3_diff_pct' in comparison:
                vol_diff = comparison['volume_mm3_diff_pct']
                status = "✓" if vol_diff < 10 else "⚠"
                print(f"  {status} Volume difference: {vol_diff:.2f}%")
            
        except Exception as e:
            print(f"  Error processing patient {patient_id}: {e}")
            continue
    
    if not comparisons:
        print("\nNo comparisons generated. Make sure predictions exist.")
        return
    
    # Create comparison dataframe
    comp_df = pd.DataFrame(comparisons)
    
    # Calculate summary statistics
    print("\n" + "="*60)
    print("AUTOMATION IMPACT SUMMARY")
    print("="*60)
    
    key_features = ['volume_mm3', 'hu_mean', 'pct_hu_lt_0', 'muscle_quality_index']
    
    for feat in key_features:
        diff_col = f'{feat}_diff_pct'
        if diff_col in comp_df.columns:
            mean_diff = comp_df[diff_col].mean()
            median_diff = comp_df[diff_col].median()
            max_diff = comp_df[diff_col].max()
            
            print(f"\n{feat}:")
            print(f"  Mean difference: {mean_diff:.2f}%")
            print(f"  Median difference: {median_diff:.2f}%")
            print(f"  Max difference: {max_diff:.2f}%")
            
            # Check if within 10% threshold
            within_10 = (comp_df[diff_col] < 10).sum()
            pct_within_10 = within_10 / len(comp_df) * 100
            print(f"  Within 10% threshold: {within_10}/{len(comp_df)} ({pct_within_10:.1f}%)")
    
    # Save comparison results
    comp_csv = results_path / 'automation_impact_comparison.csv'
    comp_df.to_csv(comp_csv, index=False)
    print(f"\n✓ Comparison results saved to: {comp_csv}")
    
    # Save summary report
    report_path = results_path / 'automation_impact_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Automation Impact Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Patients compared: {len(comp_df)}\n\n")
        
        for feat in key_features:
            diff_col = f'{feat}_diff_pct'
            if diff_col in comp_df.columns:
                mean_diff = comp_df[diff_col].mean()
                within_10 = (comp_df[diff_col] < 10).sum()
                f.write(f"{feat}:\n")
                f.write(f"  Mean difference: {mean_diff:.2f}%\n")
                f.write(f"  Within 10%: {within_10}/{len(comp_df)} ({within_10/len(comp_df)*100:.1f}%)\n\n")
        
        f.write("\nConclusion:\n")
        vol_within_10 = (comp_df['volume_mm3_diff_pct'] < 10).sum() if 'volume_mm3_diff_pct' in comp_df.columns else 0
        if vol_within_10 == len(comp_df):
            f.write("✓ All volume differences < 10% - Automation is reliable!\n")
        elif vol_within_10 >= len(comp_df) * 0.8:
            f.write("✓ Most volume differences < 10% - Automation is mostly reliable.\n")
        else:
            f.write("⚠ Some volume differences >= 10% - May need model improvement.\n")
    
    print(f"✓ Summary report saved to: {report_path}")
    
    print("\n" + "="*60)
    print("Integration completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review automation_impact_comparison.csv")
    print("2. Include results in final report")
    print("3. If volume_asym diff < 10%, automation is reliable!")


if __name__ == '__main__':
    main()
