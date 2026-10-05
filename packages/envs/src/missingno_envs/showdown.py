"""Collegamento a un server Pokémon Showdown locale tramite poke-env.

Il server va avviato a parte (vedi README):
    node pokemon-showdown start --no-security
"""

from poke_env import LocalhostServerConfiguration, ServerConfiguration

# Le random battle non richiedono squadre: ideali per la fase 0.
# Per la fase 1 si passerà a "gen1ou" con squadre vere.
DEFAULT_FORMAT = "gen1randombattle"


def local_server_config() -> ServerConfiguration:
    """Configurazione del server Showdown in esecuzione su localhost."""
    return LocalhostServerConfiguration
