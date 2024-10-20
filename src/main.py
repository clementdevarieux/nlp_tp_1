# main.py
import argparse
from src.modeling.train_model import train
from src.modeling.predict_model import predict
from src.modeling.evaluate_model import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline de classification de vidéos')
    subparsers = parser.add_subparsers(dest='command')

    # Sous-commande train
    train_parser = subparsers.add_parser('train', help='Entraîner le modèle')
    train_parser.add_argument('--input_filename', type=str, required=True, help='Chemin vers le fichier de données d\'entraînement')
    train_parser.add_argument('--model_dump_filename', type=str, required=True, help='Chemin pour sauvegarder le modèle')

    # Sous-commande predict
    predict_parser = subparsers.add_parser('predict', help='Prédire avec le modèle')
    predict_parser.add_argument('--input_filename', type=str, required=True, help='Chemin vers le fichier de données à prédire')
    predict_parser.add_argument('--model_dump_filename', type=str, required=True, help='Chemin vers le modèle sauvegardé')
    predict_parser.add_argument('--output_filename', type=str, required=True, help='Chemin pour sauvegarder les prédictions')

    # Sous-commande evaluate
    evaluate_parser = subparsers.add_parser('evaluate', help='Évaluer le modèle')
    evaluate_parser.add_argument('--input_filename', type=str, required=True, help='Chemin vers le fichier de données d\'entraînement')

    args = parser.parse_args()

    if args.command == 'train':
        train(args.input_filename, args.model_dump_filename)
    elif args.command == 'predict':
        predict(args.input_filename, args.model_dump_filename, args.output_filename)
    elif args.command == 'evaluate':
        evaluate(args.input_filename)