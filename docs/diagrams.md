# Diagrams and Design Artifacts

## Flowchart
Start -> Load Dataset -> Preprocess Text -> Extract Features -> Train Models -> Evaluate -> Save Model -> Deploy Flask App

## Architecture Diagram
User -> Flask App -> Prediction Module -> Trained Model + Vectorizer -> Prediction Result

## DFD Level 0
External User -> Fake News Detection System -> Prediction Result

## DFD Level 1
User submits article -> Flask app -> Preprocess text -> Vectorizer -> Model -> Prediction output

## Use Case Diagram
Actor: User
Use Cases: Submit article, View prediction, Review confidence score

## Sequence Diagram
User -> Flask App -> Prediction Module -> Model -> Response

## Activity Diagram
Input article -> Preprocess -> Vectorize -> Predict -> Display result

## Class Diagram
Classes: Preprocessor, Vectorizer, Classifier, EvaluationMetrics, FlaskApp

## ER Diagram
No database is required for this project; therefore, an ER diagram is not necessary.

## Testing Report
- Unit-level checks for preprocessing utilities
- Manual validation of web interface
- Model training smoke test

## Sample Output Screenshots
Screenshots can be added after running the application locally.

## Future Enhancements
- Add transformers
- Deploy with Docker
- Add API support

## Project Conclusion
The system provides a solid foundation for fake news detection and demonstrates a complete end-to-end machine learning workflow.
