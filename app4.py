import os
import pandas as pd
import joblib
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix  # Ensure this is the correct import
import ipaddress
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Dummy user database
users = {
    "1": UserMixin()
}

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# Add default user
default_user_id = "1"
default_user_username = "default_user"
default_user_password = generate_password_hash("default_password")
users[default_user_id] = User(default_user_id, default_user_username, default_user_password)

# Function to preprocess data for training and prediction
def preprocess_data(df):
    logging.debug(f'Preprocessing data: {df.head()}')
    # Convert IP addresses to integer format for machine learning
    df['src_ip'] = df['src_ip'].apply(lambda x: int(ipaddress.IPv4Address(x)))
    df['dst_ip'] = df['dst_ip'].apply(lambda x: int(ipaddress.IPv4Address(x)))
    # One-hot encode protocol and signature fields, handle missing columns
    if 'protocol' in df.columns:
        df = pd.get_dummies(df, columns=['protocol'], dummy_na=True)
    if 'signature' in df.columns:
        df = pd.get_dummies(df, columns=['signature'], dummy_na=True)
    return df

# Function to convert integers back to IP addresses for display
def convert_to_ip(df):
    df['src_ip'] = df['src_ip'].apply(lambda x: str(ipaddress.IPv4Address(x)))
    df['dst_ip'] = df['dst_ip'].apply(lambda x: str(ipaddress.IPv4Address(x)))
    return df

def train_and_save_model():
    try:
        if os.path.exists('trained_data.csv'):
            # Load training data if it exists
            df = pd.read_csv('trained_data.csv')
        else:
            if os.path.exists('network_traffic.csv'):
                # Load initial training data if it exists
                df = pd.read_csv('network_traffic.csv')
            else:
                logging.error('No training data available to train the model.')
                return "No training data available"

        # Save class distribution
        class_distribution = df['label'].value_counts()
        class_distribution.to_csv('class_distribution.csv')

        original_df = df.copy()  # Keep a copy of the original data without one-hot encoding
        
        df = preprocess_data(df)
        X = df.drop('label', axis=1)  # Features for training
        y = df['label']  # Labels for training
        
        # Ensure no NaN values in input data
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save train and test class distribution
        train_class_distribution = y_train.value_counts()
        test_class_distribution = y_test.value_counts()
        train_class_distribution.to_csv('train_class_distribution.csv')
        test_class_distribution.to_csv('test_class_distribution.csv')

        # Train the RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, 'rf_model.pkl')
        original_df.to_csv('trained_data.csv', index=False)  # Save the original data without one-hot encoding
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Save actual vs. predicted values
        actual_vs_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        actual_vs_predicted.to_csv('actual_vs_predicted.csv', index=False)

        # Explicitly use sklearn.metrics.confusion_matrix
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        logging.debug(f"Using confusion_matrix from sklearn.metrics: {sk_confusion_matrix}")

        # Calculate the confusion matrix
        cm = sk_confusion_matrix(y_test, y_pred, labels=[0, 1])  # Specify labels explicitly
        logging.debug("Confusion Matrix:")
        logging.debug(cm)

        # Convert the confusion matrix to a DataFrame for easy saving and display
        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        # Save the confusion matrix to a CSV file
        cm_df.to_csv('confusion_matrix.csv')
        
        # Calculate and save feature importances
        feature_importances = model.feature_importances_
        feature_names = X.columns
        # Create a DataFrame for feature importances
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        # Save the feature importances to a CSV file
        fi_df.to_csv('feature_importances.csv', index=False)
        
        logging.info('Model trained and saved with confusion matrix and feature importances.')
        return "Model trained and saved with confusion matrix and feature importances."
    except Exception as e:
        logging.error(f'Error in train_and_save_model: {e}')
        return str(e)

@app.route('/debug_data')
@login_required
def debug_data():
    try:
        class_distribution = pd.read_csv('class_distribution.csv') if os.path.exists('class_distribution.csv') else None
        train_class_distribution = pd.read_csv('train_class_distribution.csv') if os.path.exists('train_class_distribution.csv') else None
        test_class_distribution = pd.read_csv('test_class_distribution.csv') if os.path.exists('test_class_distribution.csv') else None
        actual_vs_predicted = pd.read_csv('actual_vs_predicted.csv') if os.path.exists('actual_vs_predicted.csv') else None

        return render_template('debug_data.html', 
                               class_distribution=class_distribution,
                               train_class_distribution=train_class_distribution,
                               test_class_distribution=test_class_distribution,
                               actual_vs_predicted=actual_vs_predicted)
    except Exception as e:
        logging.error(f'Error loading debug data: {e}')
        flash('Error loading debug data.', 'danger')
        return redirect(url_for('index'))



# Function to create a pie chart of threat breakdown
def create_pie_chart():
    if os.path.exists('trained_data.csv'):
        # Load trained data if it exists
        data = pd.read_csv('trained_data.csv')
        data = convert_to_ip(data)
        threat_count = data['label'].value_counts()
        labels = ['Non-Threat', 'Threat']
        sizes = [threat_count.get(0, 0), threat_count.get(1, 0)]
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)  # Corrected format string
        ax1.axis('equal')
        plt.savefig('static/threat_pie_chart.png')
        plt.close(fig1)  # Close the figure to avoid warnings
    else:
        logging.debug('No trained data available for pie chart.')

# Load the model if it exists, otherwise train and save it
if not os.path.exists('rf_model.pkl'):
    train_and_save_model()
else:
    model = joblib.load('rf_model.pkl')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        user_id = str(len(users) + 1)
        users[user_id] = User(user_id, username, password)
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = next((u for u in users.values() if u.username == username), None)
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    create_pie_chart()  # Generate the pie chart each time the index page is loaded
    total_events = 0
    if os.path.exists('trained_data.csv'):
        data = pd.read_csv('trained_data.csv')
        total_events = len(data)
    return render_template('index.html', total_events=total_events)


@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction_form():
    feature_importances = None  # Initialize feature_importances
    feature_names = None  # Initialize feature_names
    
    if request.method == 'POST':
        src_ip = request.form['src_ip']
        dst_ip = request.form['dst_ip']
        src_port = int(request.form['src_port'])
        dst_port = int(request.form['dst_port'])
        protocol = request.form['protocol']
        signature = request.form['signature']
        sample_weight = float(request.form['sample_weight'])
        
        # Get feature weights
        weight_src_ip = float(request.form['weight_src_ip'])
        weight_dst_ip = float(request.form['weight_dst_ip'])
        weight_src_port = float(request.form['weight_src_port'])
        weight_dst_port = float(request.form['weight_dst_port'])
        weight_protocol = float(request.form['weight_protocol'])
        weight_signature = float(request.form['weight_signature'])
        
        try:
            # Create a DataFrame for the input data
            df = pd.DataFrame([[src_ip, dst_ip, src_port, dst_port, protocol, signature]], 
                              columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'signature'])
            df = preprocess_data(df)
            
            # Apply feature weights
            df['src_ip'] *= weight_src_ip
            df['dst_ip'] *= weight_dst_ip
            df['src_port'] *= weight_src_port
            df['dst_port'] *= weight_dst_port
            if f'protocol_{protocol}' in df.columns:
                df[f'protocol_{protocol}'] *= weight_protocol
            else:
                df[f'protocol_{protocol}'] = weight_protocol
            if f'signature_{signature}' in df.columns:
                df[f'signature_{signature}'] *= weight_signature
            else:
                df[f'signature_{signature}'] = weight_signature

            df = df.reindex(columns=model.feature_names_in_, fill_value=0)  # Ensure all features match
            prediction = model.predict(df)
            prediction_proba = model.predict_proba(df)

            # Get decision path
            node_indicator, _ = model.decision_path(df)
            decision_path_dense = node_indicator.todense()

            logging.debug(f'Single prediction: {prediction[0]} for data {df}')
            
            # Get feature importances
            feature_importances = model.feature_importances_
            feature_names = df.columns.tolist()
            
            return render_template('prediction_form.html', prediction=prediction[0], 
                                   prediction_proba=prediction_proba[0], 
                                   decision_path_dense=decision_path_dense, 
                                   src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port, protocol=protocol, signature=signature,
                                   feature_importances=feature_importances, feature_names=feature_names,
                                   weight_src_ip=weight_src_ip, weight_dst_ip=weight_dst_ip, 
                                   weight_src_port=weight_src_port, weight_dst_port=weight_dst_port,
                                   weight_protocol=weight_protocol, weight_signature=weight_signature)
        except ValueError as e:
            logging.error(f'Error in prediction: {e}')
            flash(f'Error in prediction: {e}', 'danger')
            return render_template('prediction_form.html', prediction=None, 
                                   prediction_proba=None, 
                                   decision_path_dense=None, 
                                   feature_importances=feature_importances, feature_names=feature_names)
    return render_template('prediction_form.html', prediction=None, 
                           prediction_proba=None, 
                           decision_path_dense=None,
                           feature_importances=feature_importances, feature_names=feature_names)


@app.route('/save_prediction', methods=['POST'])
@login_required
def save_prediction():
    if request.method == 'POST':
        src_ip = request.form['src_ip']
        dst_ip = request.form['dst_ip']
        src_port = int(request.form['src_port'])
        dst_port = int(request.form['dst_port'])
        protocol = request.form['protocol']
        signature = request.form['signature']
        label = int(float(request.form['label']))  # Convert to float first to handle '1.0'

        try:
            # Save the prediction data to the trained data
            new_data = pd.DataFrame([[src_ip, dst_ip, src_port, dst_port, protocol, signature, label]], columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'signature', 'label'])
            if os.path.exists('trained_data.csv'):
                new_data.to_csv('trained_data.csv', mode='a', header=False, index=False)
            else:
                new_data.to_csv('trained_data.csv', mode='w', header=True, index=False)
            logging.info(f'Added new data to training set: {new_data}')
            train_and_save_model()
            flash('Prediction added to training data and model retrained!', 'success')
        except ValueError as e:
            logging.error(f'Error in saving prediction: {e}')
            flash(f'Error in saving prediction: {e}', 'danger')
        return redirect(url_for('prediction_form'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            logging.debug(f'Uploaded data: {df.head()}')
            df_preprocessed = preprocess_data(df)
            df_preprocessed = df_preprocessed.reindex(columns=model.feature_names_in_, fill_value=0)  # Ensure all features match
            logging.debug(f'Preprocessed data: {df_preprocessed.head()}')
            predictions = model.predict(df_preprocessed)

            df['prediction'] = predictions
            df = convert_to_ip(df)
            logging.debug(f'Batch predictions: {df}')
            return render_template('upload.html', tables=[df.to_html(classes='table table-striped', index=False)], titles=['Batch Predictions'], data=df)
        except Exception as e:
            logging.error(f'Error processing file upload: {e}')
            flash(f'Error processing file upload: {e}', 'danger')
            return redirect(url_for('upload_file'))
    return render_template('upload.html')

@app.route('/save_predictions', methods=['POST'])
@login_required
def save_predictions():
    if request.method == 'POST':
        try:
            data = request.form.to_dict(flat=False)
            logging.debug(f'Predictions data received: {data}')
            
            # Ensure all relevant input lists are of the same length
            relevant_keys = ['src_ip_', 'dst_ip_', 'src_port_', 'dst_port_', 'protocol_', 'signature_', 'prediction_']
            lengths = [len([key for key in data if key.startswith(rk)]) for rk in relevant_keys]
            
            if len(set(lengths)) > 1:
                raise ValueError("All input lists must have the same length")

            # Process predictions data
            for i in range(lengths[0]):
                if f'add_to_training_{i}' in data:
                    new_data = pd.DataFrame([[data[f'src_ip_{i}'][0], data[f'dst_ip_{i}'][0], int(data[f'src_port_{i}'][0]), int(data[f'dst_port_{i}'][0]), data[f'protocol_{i}'][0], data[f'signature_{i}'][0], int(float(data[f'prediction_{i}'][0]))]], 
                                            columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'signature', 'label'])
                    if os.path.exists('trained_data.csv'):
                        new_data.to_csv('trained_data.csv', mode='a', header=False, index=False)
                    else:
                        new_data.to_csv('trained_data.csv', mode='w', header=True, index=False)
                    logging.info(f'Added new data to training set: {new_data}')
            
            train_and_save_model()
            flash('Predictions saved and model retrained!', 'success')
            return redirect(url_for('upload_file'))
        except Exception as e:
            logging.error(f'Error saving predictions: {e}')
            flash(f'Error saving predictions: {e}', 'danger')
            return redirect(url_for('upload_file'))

@app.route('/trained_data')
@login_required
def trained_data():
    if os.path.exists('trained_data.csv'):
        data = pd.read_csv('trained_data.csv')
        data = convert_to_ip(data)
        return render_template('trained_data.html', data=data)
    else:
        flash('No trained data available.', 'danger')
        return redirect(url_for('index'))

@app.route('/retrain', methods=['GET', 'POST'])
@login_required
def retrain():
    try:
        if request.method == 'POST':
            train_and_save_model()
            flash('Model retrained successfully!', 'success')
            return redirect(url_for('index'))
    except Exception as e:
        logging.error(f'Error in /retrain route: {e}')
        flash(f'Error retraining model: {e}', 'danger')
        return redirect(url_for('index'))
    return render_template('retrain.html')


@app.route('/removed_data')
@login_required
def removed_data():
    # Check if the file is empty
    if os.path.getsize('removed_data.csv') == 0:
        flash('No removed data available.', 'danger')
        return redirect(url_for('index'))
    
    # Load and display removed data
    removed_data = pd.read_csv('removed_data.csv')
    removed_data = convert_to_ip(removed_data)
    return render_template('removed_data.html', tables=[removed_data.to_html(classes='table table-striped', index=False)], titles=[''])

@app.route('/add_data', methods=['GET', 'POST'])
@login_required
def add_data():
    if request.method == 'POST':
        src_ip = request.form['src_ip']
        dst_ip = request.form['dst_ip']
        src_port = int(request.form['src_port'])
        dst_port = int(request.form['dst_port'])
        protocol = request.form['protocol']
        signature = request.form['signature']
        
        try:
            # Add new data to the CSV file and retrain the model
            new_data = pd.DataFrame([[src_ip, dst_ip, src_port, dst_port, protocol, signature]], columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'signature'])
            new_data.to_csv('trained_data.csv', mode='a', header=False, index=False)
            logging.info(f'Added new data to training set: {new_data}')
            train_and_save_model()
            flash('Data added successfully!', 'success')
        except ValueError as e:
            logging.error(f'Error in adding data: {e}')
            flash(f'Error in adding data: {e}', 'danger')
        return redirect(url_for('add_data'))
    return render_template('add_data.html')

@app.route('/edit_signature/<int:index>', methods=['GET', 'POST'])
@login_required
def edit_signature(index):
    df = pd.read_csv('trained_data.csv')
    if request.method == 'POST':
        # Update the signature of the selected row
        new_signature = request.form['new_signature']
        df.at[index, 'signature'] = new_signature
        df.to_csv('trained_data.csv', index=False)
        logging.info(f'Updated signature at index {index} to {new_signature}')
        flash('Signature updated successfully!', 'success')
        return redirect(url_for('trained_data'))
    df = convert_to_ip(df)
    return render_template('edit_signature.html', index=index, current_signature=df.at[index, 'signature'])

@app.route('/delete_entry/<int:index>', methods=['GET', 'POST'])
@login_required
def delete_entry(index):
    if os.path.exists('trained_data.csv'):
        df = pd.read_csv('trained_data.csv')
        df = df.drop(index)
        df.to_csv('trained_data.csv', index=False)
        logging.info(f'Deleted entry at index {index}')
        flash('Entry deleted successfully!', 'success')
    return redirect(url_for('trained_data'))

@app.route('/edit_label/<int:index>', methods=['GET', 'POST'])
@login_required
def edit_label(index):
    try:
        df = pd.read_csv('trained_data.csv')
    except FileNotFoundError:
        flash('Training data file not found.', 'danger')
        return redirect(url_for('trained_data'))
    except Exception as e:
        logging.error(f'Error reading training data file: {e}')
        flash('An error occurred while reading the training data file.', 'danger')
        return redirect(url_for('trained_data'))

    if request.method == 'POST':
        try:
            # Update the label of the selected row
            new_label = int(request.form['new_label'])
            df.at[index, 'label'] = new_label  # Corrected assignment syntax
            df.to_csv('trained_data.csv', index=False)
            logging.info(f'Updated label at index {index} to {new_label}')
            flash('Label updated successfully!', 'success')
            return redirect(url_for('trained_data'))
        except Exception as e:
            logging.error(f'Error updating label: {e}')
            flash(f'Error updating label: {e}', 'danger')
            return redirect(url_for('edit_label', index=index))

    try:
        current_label = df.at[index, 'label']
    except KeyError:
        flash('Invalid index specified.', 'danger')
        return redirect(url_for('trained_data'))
    except Exception as e:
        logging.error(f'Error accessing label at index {index}: {e}')
        flash('An error occurred while accessing the label.', 'danger')
        return redirect(url_for('trained_data'))

    df = convert_to_ip(df)  # Assuming convert_to_ip is defined elsewhere
    return render_template('edit_label.html', index=index, current_label=current_label)
@app.route('/feature_importances')
@login_required
def feature_importances():
    try:
        if os.path.exists('feature_importances.csv'):
            data = pd.read_csv('feature_importances.csv')
            print(f"Feature Importances Data:\n{data}")  # Debug statement to log the content of the CSV
            logging.debug(f"Feature Importances Data:\n{data}")  # Log the content of the CSV
            return render_template('feature_importances.html', data=data)
        else:
            flash('Feature importances not available.', 'danger')
            return redirect(url_for('index'))
    except Exception as e:
        logging.error(f'Error loading feature importances: {e}')
        flash('Error loading feature importances.', 'danger')
        return redirect(url_for('index'))

@app.route('/show_confusion_matrix')
@login_required
def show_confusion_matrix():
    try:
        if os.path.exists('confusion_matrix.csv'):
            data = pd.read_csv('confusion_matrix.csv')
            print(f"Confusion Matrix Data:\n{data}")  # Debug statement to log the content of the CSV
            logging.debug(f"Confusion Matrix Data:\n{data}")  # Log the content of the CSV
            return render_template('confusion_matrix.html', data=data)
        else:
            flash('Confusion matrix not available.', 'danger')
            return redirect(url_for('index'))
    except Exception as e:
        logging.error(f'Error loading confusion matrix: {e}')
        flash('Error loading confusion matrix.', 'danger')
        return redirect(url_for('index'))



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
