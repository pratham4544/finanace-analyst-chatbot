
# Bhagvad Gita Conversational AI

This Streamlit application allows users to ask questions to Lord Krishna based on the text from Bhagavad Gita As It Is. The application utilizes AI-powered conversational capabilities to provide answers to user queries and insights from the text.

## Prerequisites

Before running the application, make sure you have the following dependencies installed:

- Python 3.x
- Streamlit
- PyPDF
- langchain
- pandas
- dotenv

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Run the application using the following command:

```bash
streamlit run app.py
```

4. The Streamlit app will open in your default web browser.

## Usage

1. Once the application is running, you'll see a header "Bhagvad Gita As It Is" and a subheader "Ask your question to Lord Krishan".
2. Type your question in the text input field labeled "Ask a Question to Bhagwan Krishna".
3. Press Enter or click outside the input field to submit your question.
4. Lord Krishna will provide an answer based on the context of Bhagavad Gita As It Is.

## File Structure

- `app.py`: Contains the main Streamlit application code.
- `README.md`: Instructions and information about the project.
- `requirements.txt`: List of Python dependencies required for the project.

## Additional Notes

- The application uses Google's Generative AI for embeddings and conversational responses.
- The text from Bhagavad Gita As It Is is extracted from the provided PDF file.
- Answers are provided based on contextual understanding and relevant verses from Bhagavad Gita.
- The application is configured to run with Streamlit's default settings.

## License

This project is licensed under the [MIT License](LICENSE).
```

Feel free to adjust the contents of the README.md file as needed!
