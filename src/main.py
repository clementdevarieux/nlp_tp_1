import argparse
from modeling.train_model import train
from modeling.predict_model import predict
from modeling.evaluate_model import evaluate
from datasplit import split_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline de classification de vidéos')
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train', help='Entraîner le modèle')
    train_parser.add_argument('--input_filename', type=str, required=True, help='Chemin vers le fichier de données d\'entraînement')
    train_parser.add_argument('--model_dump_filename', type=str, required=True, help='Chemin pour sauvegarder le modèle')

    predict_parser = subparsers.add_parser('predict', help='Prédire avec le modèle')
    predict_parser.add_argument('--input_filename', type=str, required=True, help='Chemin vers le fichier de données à prédire')
    predict_parser.add_argument('--model_dump_filename', type=str, required=True, help='Chemin vers le modèle sauvegardé')
    predict_parser.add_argument('--output_filename', type=str, required=True, help='Chemin pour sauvegarder les prédictions')

    evaluate_parser = subparsers.add_parser('evaluate', help='Évaluer le modèle')
    evaluate_parser.add_argument('--input_filename', type=str, required=True, help='Chemin vers le fichier de données d\'entraînement')
    evaluate_parser.add_argument('--model_dump_filename', type=str, required=True, help='Chemin vers le modèle sauvegardé')

    datasplit_parser = subparsers.add_parser('split-dataset', help="Permet de split le dataset en fonction d'un %")
    datasplit_parser.add_argument('--input_filename', type=str, required=True, help='Chemin vers le dataset à split')
    datasplit_parser.add_argument('--output_filename', type=str, required=True, help='Chemin pour sauvegarder les datasets split')
    datasplit_parser.add_argument('--split_percentage', type=float, required=True, help='le split % (en float)')

    args = parser.parse_args()

    if args.command == 'train':
        train(args.input_filename, args.model_dump_filename)
    elif args.command == 'predict':
        predict(args.input_filename, args.model_dump_filename, args.output_filename)
    elif args.command == 'evaluate':
        evaluate(args.input_filename, args.model_dump_filename)
    elif args.command == 'split-dataset':
        split_dataset(args.input_filename, args.split_percentage, args.output_filename)