pipeline {
    agent any

    environment {
        PYTHON_VERSION = "3.10"
    }

    stages {

        stage('Environment Setup') {
            steps {
                echo "=== Setting up environment ==="
                sh """
                    python3 --version
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                """
            }
        }

        stage('Pipeline Compilation') {
            steps {
                echo "=== Compiling Kubeflow Pipeline ==="
                sh """
                    . venv/bin/activate
                    python pipeline.py
                """
            }
        }

        stage('Validation / Tests') {
            steps {
                echo "=== Running tests or validations ==="
                sh """
                    echo 'Validating pipeline.yaml...'
                    test -f pipeline.yaml
                """
            }
        }
    }

    post {
        success {
            echo "Pipeline compiled and validated successfully!"
        }
        failure {
            echo "Pipeline failed!"
        }
    }
}
