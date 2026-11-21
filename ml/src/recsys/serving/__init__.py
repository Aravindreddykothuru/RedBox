"""Serving: torch-free scorers, the exported model bundle and the recommendation engine used by the API.

Every scorer works from a user's *current* history (no user ids from training), so a brand-new user is
served immediately and every new rating changes the next recommendations without retraining.
"""
