# 🎯 Student Employability Evaluation

This project is a web-based Employability Evaluation System built using Gradio. It leverages Logistic Regression and Perceptron models to assess a candidate's employability based on various soft skills.

## 🚀 Features
- User-friendly Gradio interface
- Two machine learning models: Logistic Regression and Perceptron
- Instant employability prediction
- Model training and persistent storage with pickle

## 🔑 Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/student-employability-eval.git
cd student-employability-eval
pip install -r requirements.txt
```

## 🚀 Usage
Run the app locally:

```bash
python app.py
```
Access the app at: [http://127.0.0.1:7860](http://127.0.0.1:7860)

## 📄 Demo Video


https://github.com/user-attachments/assets/8057ba8f-6568-436a-888c-90061625cadb


## 📂 Folder Structure
```
📁 student-employability-eval/
├── app.py              # Gradio App Code
├── requirements.txt    # Dependencies
├── scaler.pkl          # Scaler Model (Generated after first run)
├── logistic_regression.pkl # Logistic Regression Model (Generated after first run)
├── perceptron.pkl      # Perceptron Model (Generated after first run)
└── README.md           # Documentation
```

## 🌐 Deploy on Hugging Face Spaces
1. Go to Hugging Face Spaces
2. Click **Create New Space**
3. Select Gradio as the SDK
4. Upload `app.py` and `requirements.txt`
5. Your app will be live within minutes 🚀

## 🧐 Example Output
| Input               | Employability Prediction |
|----------------|-------------------|
| High Communication Skills, Good Mental Alertness | Employable 😊 |
| Poor Self Confidence, Low Presentation Skills | Less Employable - Work Hard! 💪 |

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit a Pull Request.

## 📌 Technologies Used
- Gradio
- Python
- Scikit-learn
- Pandas

## ⭐ Don't forget to star the repo if you like it!

