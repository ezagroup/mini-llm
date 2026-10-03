import json
import re
from pathlib import Path
from typing import List, Dict, Tuple


class SimpleTokenizer:
    """
    Un tokenizer simple et pédagogique construit depuis zéro.
    
    Approche :
    - Tokenization basée sur les mots et la ponctuation
    - Vocabulaire déterministe avec tokens spéciaux réservés
    - Encoding : texte → liste d'IDs
    - Decoding : liste d'IDs → texte approximatif
    
    Tokens spéciaux réservés (IDs fixes) :
    - 0: <PAD>   (padding)
    - 1: <UNK>   (inconnu)
    - 2: <BOS>   (début de séquence)
    - 3: <EOS>   (fin de séquence)
    """
    
    # Tokens spéciaux avec IDs réservés (déterministes)
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3,
    }
    
    def __init__(self):
        """Initialise le tokenizer avec les tokens spéciaux."""
        # token → ID mapping
        self.token2id = self.SPECIAL_TOKENS.copy()
        # ID → token mapping (inverse)
        self.id2token = {v: k for k, v in self.token2id.items()}
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Divise le texte en tokens (mots et ponctuation).
        
        Approche simple :
        - Convertir en minuscules
        - Utiliser une regex pour séparer les mots et la ponctuation
        - Supprimer les espaces vides
        
        Args:
            text: Texte à tokenizer
            
        Returns:
            Liste de tokens
        """
        # Convertir en minuscules pour la normalisation
        text = text.lower()
        
        # Regex : sépare les mots et la ponctuation
        # \w+ : séquence de caractères alphanumériques
        # [^\w\s] : caractères de ponctuation (pas alphanumériques ni espaces)
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        return tokens
    
    def build_vocab(self, text: str) -> None:
        """
        Construit le vocabulaire à partir d'un texte.
        
        Le vocabulaire est déterministe :
        - Les tokens spéciaux gardent leurs IDs
        - Les nouveaux tokens reçoivent des IDs séquentiels à partir de 4
        
        Args:
            text: Texte d'entraînement pour construire le vocabulaire
        """
        # Tokenizer le texte
        tokens = self._tokenize(text)
        
        # Obtenir l'ensemble unique des tokens (en préservant l'ordre d'apparition)
        # Utiliser un dict pour préserver l'ordre en Python 3.7+
        unique_tokens = dict.fromkeys(tokens)
        
        # Ajouter les nouveaux tokens au vocabulaire
        # Les IDs commencent à 4 (après les tokens spéciaux)
        next_id = len(self.token2id)
        
        for token in unique_tokens.keys():
            if token not in self.token2id:
                self.token2id[token] = next_id
                self.id2token[next_id] = token
                next_id += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Encode du texte en liste d'IDs.
        
        Le vocabulaire ne doit jamais être modifié pendant l'encodage.
        Les tokens inconnus sont remplacés par <UNK> (ID 1).
        
        Args:
            text: Texte à encoder
            
        Returns:
            Liste d'identifiants (IDs)
        """
        tokens = self._tokenize(text)
        ids = []
        
        for token in tokens:
            # Si le token est dans le vocabulaire, utiliser son ID
            # Sinon, utiliser l'ID de <UNK>
            token_id = self.token2id.get(token, self.token2id['<UNK>'])
            ids.append(token_id)
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Décode une liste d'IDs en texte (approximativement).
        
        Approche simple :
        - Convertir chaque ID en token
        - Joindre les tokens avec des espaces
        - Gérer les tokens spéciaux (les masquer ou les afficher)
        
        Args:
            ids: Liste d'identifiants à décoder
            
        Returns:
            Texte reconstruit (approximatif)
        """
        tokens = []
        
        for token_id in ids:
            # Récupérer le token associé à cet ID
            token = self.id2token.get(token_id, '<UNK>')
            tokens.append(token)
        
        # Joindre les tokens avec des espaces
        # (Note : cela ne reconstitue pas exactement la ponctuation)
        text = ' '.join(tokens)
        
        # Nettoyer les tokens spéciaux de manière basique
        # Supprimer les espaces avant la ponctuation
        text = re.sub(r'\s+([^\w\s])', r'\1', text)
        
        return text
    
    def vocab_size(self) -> int:
        """Retourne la taille du vocabulaire."""
        return len(self.token2id)
    
    def save_vocab(self, filepath: str) -> None:
        """
        Sauvegarde le vocabulaire dans un fichier JSON.
        
        Format : {"token": id, "token": id, ...}
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, filepath: str) -> None:
        """
        Recharge un vocabulaire depuis un fichier JSON.
        
        Les IDs doivent rester strictement identiques après rechargement.
        
        Args:
            filepath: Chemin du fichier de vocabulaire
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.token2id = json.load(f)
        
        # Reconstruire le mapping inverse
        self.id2token = {int(v): k for k, v in self.token2id.items()}
    
    def get_token(self, token_id: int) -> str:
        """Retourne le token associé à un ID."""
        return self.id2token.get(token_id, '<UNK>')
    
    def get_id(self, token: str) -> int:
        """Retourne l'ID associé à un token."""
        return self.token2id.get(token, self.token2id['<UNK>'])
