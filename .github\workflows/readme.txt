# Citi Bike Analysis Project

A comprehensive machine learning project for analyzing Citi Bike data with automated pipelines and workflows.

## 🚴‍♀️ Project Overview

This project provides an end-to-end solution for analyzing Citi Bike usage patterns using machine learning techniques. It includes data processing pipelines, model training workflows, and inference capabilities to derive insights from bike-sharing data.

## 📁 Project Structure

```
citi_bike/
├── .github/
│   └── workflows/           # GitHub Actions CI/CD workflows
│       ├── feature_pipeline.yaml
│       ├── inference_pipeline.yaml
│       ├── model_training_pipeline.yaml
│       └── readme.txt
├── models/                  # Trained model artifacts and configurations
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── src/                    # Source code and utilities
├── workflows/              # Additional workflow configurations
├── requirements.txt        # Python dependencies
├── requirements_feature_pipeline.txt    # Feature pipeline specific dependencies
├── requirements_with_version.txt        # Pinned version dependencies
└── README.md              # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/siddhyaaddy/citi_bike.git
   cd citi_bike
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # For general use
   pip install -r requirements.txt
   
   # For feature pipeline development
   pip install -r requirements_feature_pipeline.txt
   
   # For exact version reproducibility
   pip install -r requirements_with_version.txt
   ```

## 🚀 Usage

### Data Processing
The project includes automated feature engineering pipelines that process raw Citi Bike data into ML-ready features.

### Model Training
Machine learning models are trained using the configured pipelines in the `.github/workflows/` directory.

### Inference
Run inference on new data using the trained models stored in the `models/` directory.

## 📊 Features

- **Automated Pipelines**: GitHub Actions workflows for continuous integration
- **Feature Engineering**: Robust data preprocessing and feature extraction
- **Model Training**: Scalable machine learning model training pipeline
- **Inference Pipeline**: Production-ready model serving capabilities
- **Exploratory Analysis**: Jupyter notebooks for data exploration

## 🔧 Development

### Running Locally
1. Ensure all dependencies are installed
2. Explore data using notebooks in the `notebooks/` directory
3. Modify source code in the `src/` directory
4. Test changes using the workflow configurations

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure pipelines pass
5. Submit a pull request

## 📋 Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`
- Additional pipeline-specific requirements in respective files

## 🔄 Workflows

The project uses GitHub Actions for automation:
- **Feature Pipeline**: Automated feature engineering
- **Model Training**: Scheduled model retraining
- **Inference Pipeline**: Model deployment and serving

## 📈 Project Status

- **Language Distribution**: 99.7% Jupyter Notebook, 0.3% Python
- **Last Updated**: 3 months ago
- **Commits**: 19 total commits

## 📝 License

This project is available under the MIT License. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or suggestions, please open an issue or contact the maintainer.

---

*This project analyzes Citi Bike usage patterns to provide insights into urban mobility and bike-sharing optimization.*
