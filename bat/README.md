# Batch Scripts for MLQCD-SV

This folder contains Windows batch files for running various analysis tasks.

## Available Scripts

### **interactive_analysis.bat** ⭐ (Recommended)
Interactive menu-driven analysis tool. Select individual datasets and models.
```cmd
bat\interactive_analysis.bat
```

### **run_spectral_analysis.bat**
Run batch analysis on all 7 datasets with configured models.
```cmd
bat\run_spectral_analysis.bat
```

### **run_combine.bat**
Combine spectral fit results from multiple experiments into one CSV file.
```cmd
bat\run_combine.bat
```

### **extract_parameters.bat**
Extract spectral fit parameters from result directories.
```cmd
bat\extract_parameters.bat
```

## Usage

1. **Double-click** any `.bat` file in Windows Explorer, or
2. **Run from command prompt**:
   ```cmd
   cd C:\path\to\MLQCD-SV
   bat\interactive_analysis.bat
   ```

## Notes

- All batch files automatically handle virtual environment activation
- Scripts change to the project root directory before running
- Results are saved to the main project directories
- Use `interactive_analysis.bat` for most flexible analysis options