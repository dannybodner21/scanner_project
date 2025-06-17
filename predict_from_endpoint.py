from google.cloud import aiplatform
import pandas as pd
from sklearn.metrics import classification_report

def predict_sample_from_endpoint(project, region, endpoint_id, test_csv_path, sample_size=10000):
    aiplatform.init(project=project, location=region)
    endpoint = aiplatform.Endpoint(endpoint_id)

    df = pd.read_csv(test_csv_path)
    sample_df = df.sample(n=sample_size, random_state=42)

    X = sample_df.drop(columns=['label', 'coin'], errors='ignore')

    predictions = []
    batch_size = 100

    for i in range(0, len(X), batch_size):
        batch = X.iloc[i:i+batch_size].astype(str).to_dict(orient='records')
        response = endpoint.predict(instances=batch)
        preds = [pred['scores'][0] for pred in response.predictions]  # <-- fix here
        predictions.extend(preds)




    sample_df['predicted_prob'] = predictions
    sample_df['predicted_label'] = sample_df['predicted_prob'].apply(lambda p: 1 if p >= 0.7 else 0)


    total_trades = sample_df['predicted_label'].sum()
    correct_trades = ((sample_df['predicted_label'] == 1) & (sample_df['label'] == 1)).sum()
    accuracy_of_trades = correct_trades / total_trades if total_trades > 0 else 0

    print(f"Sample size: {sample_size}")
    print(f"Total trades taken (predicted long entries): {total_trades}")
    print(f"Number of correct trades (true positives): {correct_trades}")
    print(f"Accuracy on trades taken: {accuracy_of_trades:.2%}\n")

    print("Full classification report:")
    print(classification_report(sample_df['label'], sample_df['predicted_label']))

if __name__ == "__main__":
    PROJECT_ID = 'healthy-mark-446922-p8'
    REGION = 'us-central1'
    ENDPOINT_ID = '4855915038747131904'
    TEST_CSV_PATH = 'long_model_2025_test_data.csv'

    predict_sample_from_endpoint(PROJECT_ID, REGION, ENDPOINT_ID, TEST_CSV_PATH, sample_size=10000)
