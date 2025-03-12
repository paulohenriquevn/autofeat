"""
Utilitários para o pacote CAFE.
"""

import logging
from typing import Optional, Dict, Any, Union


def setup_logger(name: str, level: Union[int, str] = logging.INFO) -> logging.Logger:
    """
    Configura e retorna um logger com o nome e nível especificados.
    
    Args:
        name: Nome do logger
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_verbosity_level(verbosity: int) -> int:
    """
    Converte o nível de verbosidade numérico para o nível de log equivalente.
    
    Args:
        verbosity: Nível de verbosidade (0=WARNING, 1=INFO, 2=DEBUG)
        
    Returns:
        Nível de log correspondente
    """
    levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    return levels.get(verbosity, logging.INFO)


def update_config(default_config: Dict[str, Any], user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Atualiza uma configuração padrão com os valores fornecidos pelo usuário.
    
    Args:
        default_config: Configuração padrão
        user_config: Configuração do usuário (opcional)
        
    Returns:
        Configuração atualizada
    """
    if user_config is None:
        return default_config.copy()
    
    result = default_config.copy()
    result.update(user_config)
    return result