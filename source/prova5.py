@cog.input(
    "sleep_type",
    type=str,
    default="consciuos",
    options=["consciuos", "unconscious"],
    deck_changer=True
    help="Type of AI dream. 'conscious' for shorter dreams with all UI controls, 'unconsciuous' for longer but more unpredictable outputs",
)
@cog.input(
    "memories",
    type=str,
    default="all",
    deck="consciuos"
    help="Type of sound memories that can occur in the dream (list of comme-divided items). Options: all, africanPercs, ambient1, buchla, buchla2, classical, classical2, guitarAcoustic, guitarBaroque, jazz, organ, percsWar, percussions, pianoChill, pianoDreamy, pianoSmooth, airport, birdsStreet, forest, library, mixed, office, rain, sea, train, wind",
)
@cog.input(
    "unconscious_dream_length",
    type=float,
    default=3,
    deck="unconscious"
    help="Approximative length in minutes of the soundfile to generate (only for unconscious dream type)",
)
