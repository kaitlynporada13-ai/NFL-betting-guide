"""Player prop prediction models."""
from models.player_props.passing_yards import PassingYardsModel
from models.player_props.rushing_yards import RushingYardsModel
from models.player_props.receiving_yards import ReceivingYardsModel
from models.player_props.receptions import ReceptionsModel
from models.player_props.touchdowns import AnytimeTDModel

ALL_PROP_MODELS = {
    "passing_yards": PassingYardsModel,
    "rushing_yards": RushingYardsModel,
    "receiving_yards": ReceivingYardsModel,
    "receptions": ReceptionsModel,
    "touchdowns_anytime": AnytimeTDModel,
}
