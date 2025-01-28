import pytest
from modules.knowledge_graph.conflict_resolution import ConflictResolver

@pytest.fixture
def config():
    return {
        "knowledge_graph": {
            "conflict_policy": "highest_confidence_wins",
            "domain_trust_scores": {
                "wikipedia.org": 0.9,
                "gov": 0.8,
                "edu": 0.8,
            },
            "bias_overrides": {
                "official_sources": ["who.int", "cdc.gov"],
                "domain_boost": {
                    "nature.com": 1.8,
                }
            }
        }
    }

def test_official_source_overrides_higher_confidence(config):
    resolver = ConflictResolver(config)
    
    # Test that WHO (official) overrides higher confidence unofficial
    result = resolver.resolve_conflict(
        existing_edge={
            "confidence": 0.9,
            "source": "example.com"
        },
        new_edge={
            "confidence": 0.7,
            "source": "who.int/example"
        }
    )
    
    assert result.action == "overwrite"
    assert result.final_confidence > 1.0  # Should be boosted

def test_official_truth_protection(config):
    resolver = ConflictResolver(config)
    
    # Test that official_truth is protected from non-official sources
    result = resolver.resolve_conflict(
        existing_edge={
            "confidence": 0.7,
            "source": "cdc.gov",
            "official_truth": True
        },
        new_edge={
            "confidence": 0.9,
            "source": "example.com"
        }
    )
    
    assert result.action == "keep_existing"

def test_domain_boost_applied(config):
    resolver = ConflictResolver(config)
    
    result = resolver.resolve_conflict(
        existing_edge={
            "confidence": 0.7,
            "source": "example.com"
        },
        new_edge={
            "confidence": 0.6,
            "source": "nature.com/article"
        }
    )
    
    assert result.action == "overwrite"  # Should win due to domain boost 

def test_multiple_domain_boosts(config):
    """Test that multiple domain boosts don't stack inappropriately"""
    resolver = ConflictResolver(config)
    
    result = resolver.resolve_conflict(
        existing_edge={
            "confidence": 0.5,
            "source": "example.com"
        },
        new_edge={
            "confidence": 0.5,
            "source": "nature.com.science.org"  # Should only apply highest boost
        }
    )
    
    # Should use highest boost (1.8) not multiply both
    assert result.final_confidence <= 0.5 * 1.8

def test_invalid_source_handling(config):
    """Test that invalid sources don't crash the resolver"""
    resolver = ConflictResolver(config)
    
    result = resolver.resolve_conflict(
        existing_edge={
            "confidence": 0.5,
            "source": None
        },
        new_edge={
            "confidence": 0.5,
            "source": "invalid://url"
        }
    )
    
    assert result.action in ["keep_existing", "overwrite"]
    assert 0 <= result.final_confidence <= 1.0 