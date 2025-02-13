import argparse
from domain_adaptation.run_domain_adaptation import run_domain
from multilingual.run_multilingual import run_multilingual
from ner_preprocessing.run_ner_preprocessing import run_ner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', choices=['domain', 'multilingual', 'ner'], required=True)
    args = parser.parse_args()

    if args.approach == 'domain':
        run_domain_adaptation()
    elif args.approach == 'multilingual':
        run_multilingual()
    elif args.approach == 'ner':
        run_ner()

if __name__ == "__main__":
    main()
