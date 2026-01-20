from .encoder_selector import ENCODER_REGISTRY, _resolve_encoder, get_loc_encoder
import unittest
from ..location_encoders.GridCellSpatialRelationLocationEncoder import GridCellSpatialRelationLocationEncoder
from ..location_encoders.XYZSpatialRelationLocationEncoder import XYZSpatialRelationLocationEncoder
import torch

# To run the tests: Go to the directory containing the TorchSpatial package and run this: (this is so that the relative imports can work)
# python -m TorchSpatial.modules.test_encoder_selector

class test_encoder_selector(unittest.TestCase):

    def test_ENCODER_REGISTRY_1(self):
        self.assertEqual(
            ENCODER_REGISTRY['xyz'], 
            ("..location_encoders.XYZSpatialRelationLocationEncoder", "XYZSpatialRelationLocationEncoder")
        )

    def test_ENCODER_REGISTRY_2(self):
        self.assertEqual(
            ENCODER_REGISTRY["Space2Vec-grid"], 
            ("..location_encoders.GridCellSpatialRelationLocationEncoder", "GridCellSpatialRelationLocationEncoder")
        )

    def test__resolve_encoder_1(self):
        self.assertEqual(
            _resolve_encoder('xyz'), 
            XYZSpatialRelationLocationEncoder
        )
    
    def test__resolve_encoder_2(self):
        self.assertEqual(
            _resolve_encoder("Space2Vec-grid"), 
            GridCellSpatialRelationLocationEncoder
        )

    def test_get_loc_encoder_1(self):
        self.assertEqual(
            (get_loc_encoder("xyz", overrides = {"spa_embed_dim": 25})).spa_embed_dim, 
            25
        )
        self.assertEqual(
            XYZSpatialRelationLocationEncoder(spa_embed_dim = 25).spa_embed_dim,
            25
        )

    def test_get_loc_encoder_2(self):
        self.assertEqual(
            get_loc_encoder("Space2Vec-grid").max_radius, 
            10000
        )
        self.assertEqual(
            GridCellSpatialRelationLocationEncoder().max_radius,
            10000
        )
        self.assertEqual(
            get_loc_encoder("Space2Vec-grid").min_radius, 
            10
        )
        self.assertEqual(
            GridCellSpatialRelationLocationEncoder().min_radius,
            10
        )

    def test_get_loc_encoder_3(self):
        self.assertEqual(
            (get_loc_encoder("xyz", overrides = {"spa_embed_dim": 5})).spa_embed_dim, 
            5
        )
        self.assertEqual(
            XYZSpatialRelationLocationEncoder(spa_embed_dim = 5).spa_embed_dim,
            5
        )

    def test_get_loc_encoder_4(self):
        self.assertEqual(
            get_loc_encoder("Space2Vec-grid", overrides = {}).max_radius, 
            10000
        )
        self.assertEqual(
            GridCellSpatialRelationLocationEncoder().max_radius,
            10000
        )
        self.assertEqual(
            get_loc_encoder("Space2Vec-grid", overrides = {}).min_radius, 
            10
        )
        self.assertEqual(
            GridCellSpatialRelationLocationEncoder().min_radius,
            10
        )

    def test_get_loc_encoder_5(self):
        self.assertEqual(
            get_loc_encoder("Space2Vec-grid", overrides = {"min_radius":10}).max_radius, 
            10000
        )
        self.assertEqual(
            GridCellSpatialRelationLocationEncoder().max_radius,
            10000
        )
        self.assertEqual(
            get_loc_encoder("Space2Vec-grid", overrides = {"min_radius":10}).min_radius, 
            10
        )
        self.assertEqual(
            GridCellSpatialRelationLocationEncoder().min_radius,
            10
        )

    def test_get_loc_encoder_6(self):
        self.assertEqual(
            get_loc_encoder("Space2Vec-grid", overrides = {"min_radius":15}).max_radius, 
            10000
        )
        self.assertEqual(
            GridCellSpatialRelationLocationEncoder(min_radius = 15).max_radius,
            10000
        )
        self.assertEqual(
            get_loc_encoder("Space2Vec-grid", overrides = {"min_radius":15}).min_radius, 
            15
        )
        self.assertEqual(
            GridCellSpatialRelationLocationEncoder(min_radius = 15).min_radius,
            15
        )

    def test_get_loc_encoder_7(self):
        self.assertEqual(
            get_loc_encoder("Space2Vec-grid", overrides = {"max_radius":7800, "min_radius":15}).max_radius, 
            7800
        )
        self.assertEqual(
            GridCellSpatialRelationLocationEncoder(max_radius = 7800, min_radius = 15).max_radius,
            7800
        )
        self.assertEqual(
            get_loc_encoder("Space2Vec-grid", overrides = {"max_radius":7800, "min_radius":15}).min_radius, 
            15
        )
        self.assertEqual(
            GridCellSpatialRelationLocationEncoder(max_radius = 7800, min_radius = 15).min_radius,
            15
        )

if __name__ == "__main__":
   unittest.main()