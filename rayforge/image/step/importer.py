import logging
import tempfile
import os
import math
from typing import Optional, List, Tuple
from pathlib import Path

# --- Dependency Check ---
# We assume false until proven true to catch DLL errors during import
CQ_AVAILABLE = False
try:
    import cadquery as cq
    from cadquery import Workplane, Vector
    CQ_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"CadQuery import failed: {e}")
except Exception as e:
    logging.getLogger(__name__).warning(f"CadQuery crashed on load: {e}")

from ...core.geo import Geometry
from ...core.workpiece import WorkPiece
from ...core.source_asset import SourceAsset
from ...core.source_asset_segment import SourceAssetSegment
from ...core.vectorization_spec import PassthroughSpec
from ...core.matrix import Matrix
from ..base_importer import Importer, ImportPayload
# Corrected import for the renderer instance
from ..svg.renderer import SVG_RENDERER

logger = logging.getLogger(__name__)

class StepImporter(Importer):
    label = "STEP (Auto-Align)"
    mime_types = ("application/step", "model/step", "text/plain") 
    extensions = (".step", ".stp")

    def get_doc_items(self, vectorization_spec=None) -> Optional[ImportPayload]:
        if not CQ_AVAILABLE:
            logger.error("Cannot import STEP: CadQuery library is missing or broken.")
            return None

        # 1. Write bytes to temp file (CadQuery requires a file path)
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
            tmp.write(self.raw_data)
            tmp_path = tmp.name

        try:
            logger.info(f"Loading STEP file: {self.source_file}")
            model = cq.importers.importStep(tmp_path)
            
            # --- Robust Object Detection ---
            # Some files are Solids, some are Shells (surfaces), some are loose Faces.
            # We try to grab the most complex object available.
            target_objects = model.solids().vals()
            if not target_objects:
                logger.info("No solids found, looking for shells...")
                target_objects = model.shells().vals()
            if not target_objects:
                logger.info("No shells found, looking for faces...")
                target_objects = model.faces().vals()
            
            if not target_objects:
                logger.error("STEP file contains no recognized geometry (Solids, Shells, or Faces).")
                return None

            primary_object = target_objects[0]

            # --- Projection Logic ---
            projection = None
            
            # Attempt 1: Auto-Align to largest flat face
            try:
                # Find all planar faces in the object
                # Note: We create a new Workplane wrapper around the object to use selectors
                wp_wrapper = cq.Workplane(obj=primary_object)
                faces = wp_wrapper.faces("%PLANE").vals()
                
                if faces:
                    # Sort by area, largest first
                    faces.sort(key=lambda f: f.Area(), reverse=True)
                    largest_face = faces[0]
                    
                    # Create a workplane aligned to this face
                    aligned_wp = cq.Workplane(obj=largest_face).workplane()
                    
                    # Add the original object to this aligned view and project it
                    projection = aligned_wp.add(primary_object).toPending()
                    logger.info("Success: Aligned to largest face.")
            except Exception as e:
                logger.warning(f"Auto-align failed ({e}). Falling back to Top-Down view.")

            # Attempt 2: Fallback to simple Top-Down projection if alignment skipped or failed
            if projection is None:
                projection = model.toPending()

            # --- Geometry Conversion ---
            geo = Geometry()
            edge_count = 0
            
            # .vals() gives us the OCCT Wire objects
            wires = projection.vals()
            if not wires:
                logger.error("Projection resulted in no geometry.")
                return None

            for wire in wires:
                edges = wire.Edges()
                if not edges: continue

                # Move to start of loop
                start_p = edges[0].startPoint()
                geo.move_to(start_p.x, start_p.y)

                for edge in edges:
                    try:
                        # Discretize: Convert CAD curve to line segments
                        # tolerance=0.05mm
                        points = edge.discretize(tolerance=0.05)
                        if not points: continue
                            
                        # Line to all points (skipping first to avoid dupes)
                        for p in points[1:]:
                            geo.line_to(p.x, p.y)
                        edge_count += 1
                    except Exception:
                        continue

            if geo.is_empty():
                logger.error("Geometry extraction failed (result was empty).")
                return None

            logger.info(f"Imported {edge_count} edges.")

            # --- Normalization & Packaging ---
            min_x, min_y, max_x, max_y = geo.rect()
            width = max_x - min_x
            height = max_y - min_y

            source = SourceAsset(
                source_file=self.source_file,
                original_data=self.raw_data,
                renderer=SVG_RENDERER, 
                metadata={"is_vector": True, "natural_size": (width, height)},
            )
            source.width_mm = width
            source.height_mm = height

            # Normalize geometry to 0..1 range and flip Y
            norm_geo = geo.copy()
            norm_geo.transform(Matrix.translation(-min_x, -min_y).to_4x4_numpy())
            
            if width > 0 and height > 0:
                norm_geo.transform(Matrix.scale(1.0/width, 1.0/height).to_4x4_numpy())
            
            # Flip Y (Rayforge internal standard)
            norm_geo.transform((Matrix.translation(0, 1) @ Matrix.scale(1, -1)).to_4x4_numpy())

            gen_config = SourceAssetSegment(
                source_asset_uid=source.uid,
                segment_mask_geometry=norm_geo,
                vectorization_spec=PassthroughSpec(),
                width_mm=width,
                height_mm=height,
            )

            wp = WorkPiece(
                name=self.source_file.stem,
                source_segment=gen_config,
            )
            wp.matrix = Matrix.translation(min_x, min_y) @ Matrix.scale(width, height)

            return ImportPayload(source=source, items=[wp])

        except Exception as e:
            logger.exception("Critical error during STEP import")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass