from typing import List, Dict, Any

# Define a type for the audio file paths
AudioFilePath = str

# Define a type for the transcription result
TranscriptionResult = Dict[str, Any]

# Define a type for the grammar correction result
GrammarCorrectionResult = str

# Define a type for the pipeline input
PipelineInput = List[AudioFilePath]

# Define a type for the pipeline output
PipelineOutput = List[GrammarCorrectionResult]