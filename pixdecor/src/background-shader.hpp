#ifndef BACKGROUND_SHADER_HPP
#define BACKGROUND_SHADER_HPP

#include <wayfire/scene.hpp>
#include <wayfire/opengl.hpp>
#include <wayfire/scene-render.hpp>
#include <wayfire/toplevel-view.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/vec4.hpp>

namespace wf
{
class background_shader_node_t : public wf::scene::node_t
{
private:
    wayfire_toplevel_view view;
    OpenGL::program_t shader_program;
    bool shader_initialized = false;
    float animation_time = 0.0f;

    const char* vertex_shader = R"(
#version 100
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
uniform mat4 mvp;

void main() {
    v_texcoord = texcoord;
    gl_Position = mvp * vec4(position, 0.0, 1.0);
}
)";

const char* fragment_shader = R"(
#version 100
#extension GL_OES_standard_derivatives : enable
precision mediump float;
varying vec2 v_texcoord;
uniform vec2 resolution;
uniform float time;
uniform vec4 color;

void main() {
    vec2 uv = v_texcoord;
    // p is in a coordinate system centered at (0,0) and corrected for aspect ratio.
    // The X-axis ranges from [-aspect, aspect] and Y-axis from [-1, 1].
    vec2 p = (uv * 2.0 - 1.0) * vec2(resolution.x / resolution.y, 1.0);
    
    // --- Rounded Corner Calculation ---
    // Define the size of the box and the corner radius in p's coordinate space.
    vec2 box_half_size = vec2(resolution.x / resolution.y, 1.0);
    float radius = 0.2; // Larger value = more rounded corners.

    // Calculate signed distance from the rounded rectangle shape.
    // This is a standard SDF formula for a rounded box.
    vec2 q = abs(p) - (box_half_size - radius);
    float dist = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - radius;

    // Create a smooth alpha mask from the distance for anti-aliasing.
    // fwidth(dist) estimates how much the distance changes between pixels.
    float aa = fwidth(dist) * 0.5;
    float mask = smoothstep(aa, -aa, dist);

    // --- Original Plasma Calculation ---
    float t = time * 0.5;
    
    float wave1 = sin(p.x * 10.0 + t);
    float wave2 = sin(p.y * 10.0 + t * 1.3);
    float wave3 = sin((p.x + p.y) * 8.0 + t * 0.8);
    float wave4 = sin(length(p) * 12.0 - t * 2.0);
    
    float plasma = (wave1 + wave2 + wave3 + wave4) * 0.25;
    
    vec3 col1 = vec3(0.5, 0.0, 1.0); // Purple
    vec3 col2 = vec3(0.0, 0.5, 1.0); // Cyan
    vec3 col3 = vec3(1.0, 0.3, 0.5); // Pink
    
    vec3 finalColor = mix(col1, col2, sin(plasma * 3.14159 + t) * 0.5 + 0.5);
    finalColor = mix(finalColor, col3, cos(plasma * 2.0 - t * 0.5) * 0.5 + 0.5);
    
    float glow = 1.0 - length(p) * 0.5;
    finalColor += vec3(0.2, 0.1, 0.3) * glow;
    
    // --- Final Color Output ---
    // Apply the rounded corner mask to the final alpha.
    gl_FragColor = vec4(finalColor * color.rgb, color.a * 0.7 * mask);
}
)";

/*
const char* fragment_shader = R"(
#version 100
#extension GL_OES_standard_derivatives : enable
precision mediump float;
varying vec2 v_texcoord;
uniform vec2 resolution;
uniform float time;
uniform vec4 color;

// --- Glowing Hexagonal Grid Pattern ---
// This function calculates the color for a glowing hex grid at a given point 'p'.
vec3 createHexGrid(vec2 p) {
    // Scale the coordinate system to control the size of the hexagons.
    p *= 8.0;

    // A common technique for creating a hexagonal grid is to overlay three sets of lines,
    // each rotated by 120 degrees. We calculate the distance to the nearest line in each set
    // and take the minimum of these distances.

    // Define the 120-degree rotation matrix (cos(120), -sin(120), etc.)
    mat2 rotation = mat2(-0.5, -0.866025, 0.866025, -0.5);

    // --- First set of lines (no rotation) ---
    // fract(p.y * 0.5) gives a sawtooth wave. Taking abs(... - 0.5) turns it into a triangle wave,
    // which represents the distance from the center of a band.
    float d = abs(fract(p.y * 0.5) - 0.5);

    // --- Second set of lines (rotated 120 degrees) ---
    p = rotation * p;
    d = min(d, abs(fract(p.y * 0.5) - 0.5));

    // --- Third set of lines (rotated another 120 degrees) ---
    p = rotation * p;
    d = min(d, abs(fract(p.y * 0.5) - 0.5));

    // 'd' now holds a value that is ~0 at the center of a hex line and 0.25 at the center of a hex cell.

    // --- Create the Glow ---
    // Use smoothstep to create a smooth falloff from the line.
    // pow() is used to sharpen the falloff, making it look more like a glow.
    float glow = pow(1.0 - smoothstep(0.0, 0.15, d), 6.0);
    
    // Add a second, wider, dimmer glow for an aura effect.
    float aura = pow(1.0 - smoothstep(0.0, 0.4, d), 4.0);

    // Animate the brightness over time to make the grid pulse.
    float pulse = 0.7 + 0.3 * sin(time * 2.0);
    
    // Combine the core glow and the aura, using the input color.
    vec3 finalGlow = color.rgb * glow + color.rgb * 0.3 * aura;
    
    return finalGlow * pulse;
}

void main() {
    vec2 uv = v_texcoord;
    // p is in a coordinate system centered at (0,0) and corrected for aspect ratio.
    vec2 p = (uv * 2.0 - 1.0) * vec2(resolution.x / resolution.y, 1.0);
    
    // --- Rounded Corner Calculation (from original shader) ---
    vec2 box_half_size = vec2(resolution.x / resolution.y, 1.0);
    float radius = 0.2;
    vec2 q = abs(p) - (box_half_size - radius);
    float dist = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    float aa = fwidth(dist) * 0.5;
    float mask = smoothstep(aa, -aa, dist);

    // --- Pattern Calculation ---
    // Get the color of the glowing hex pattern.
    vec3 col = createHexGrid(p);

    // --- Final Color Output ---
    // Add a very subtle, dark background color.
    vec3 backgroundColor = vec3(0.01, 0.0, 0.02);
    // Apply the rounded corner mask to the final alpha.
    gl_FragColor = vec4(col + backgroundColor, color.a * mask);
}
)";*/
    void init_shader()
    {
        if (shader_initialized) return;
        
        shader_program.set_simple(
            OpenGL::compile_program(vertex_shader, fragment_shader));
        shader_initialized = true;
    }

public:
    background_shader_node_t(wayfire_toplevel_view view) : node_t(false), view(view)
    {
    }

    ~background_shader_node_t()
    {
        shader_program.free_resources();
    }

    std::optional<wf::scene::input_node_t> find_node_at(const wf::pointf_t& at) override
    {
        return {};
    }

wf::geometry_t get_bounding_box() override
{
    if (!view || !view->is_mapped()) return {0, 0, 0, 0};

    auto full_geo = view->get_geometry();
    auto margins = view->toplevel()->pending().margins;

    const int max_layer = 20;
    const float spacing_per_layer = 4.0f;
    const int max_spacing = static_cast<int>(max_layer * spacing_per_layer);

    // This is the correct implementation.
    return wf::geometry_t{
        -margins.left - max_spacing,
        -margins.top - max_spacing,
        full_geo.width + 2 * max_spacing,
        full_geo.height + 2 * max_spacing
    };
}

    void gen_render_instances(std::vector<scene::render_instance_uptr>& instances,
        scene::damage_callback push_damage, wf::output_t *output) override
    {
        instances.push_back(std::make_unique<shader_render_instance_t>(this, push_damage));
    }

    class shader_render_instance_t : public scene::render_instance_t
    {
        background_shader_node_t *self;
        scene::damage_callback push_damage;

    public:
        shader_render_instance_t(background_shader_node_t *self, scene::damage_callback push_damage)
            : self(self), push_damage(push_damage)
        {}

        void schedule_instructions(std::vector<scene::render_instruction_t>& instructions,
            const wf::render_target_t& target, wf::region_t& damage) override
        {
            auto local_bbox = self->get_bounding_box();
            auto global_origin = self->to_global({(double)local_bbox.x, (double)local_bbox.y});
            
            wf::geometry_t our_region{
                (int)global_origin.x,
                (int)global_origin.y,
                local_bbox.width,
                local_bbox.height
            };
            
            wf::region_t our_damage = damage & our_region;
            if (!our_damage.empty())
            {
                instructions.push_back(scene::render_instruction_t{
                    .instance = this,
                    .target = target,
                    .damage = std::move(our_damage),
                });
            }
        }

    // Located inside the shader_render_instance_t class

// Located inside the shader_render_instance_t class

// Located inside the shader_render_instance_t class

// Located inside the shader_render_instance_t class

void render(const wf::scene::render_instruction_t& data) override
{
    if (!self->view || !self->view->is_mapped()) return;

    self->init_shader();

    auto full_geo = self->view->get_geometry();
    auto margins = self->view->toplevel()->pending().margins;

    const auto& target = data.target;
    int fb_width  = target.geometry.width;
    int fb_height = target.geometry.height;
    auto ortho = glm::ortho(0.0f, (float)fb_width, (float)fb_height, 0.0f);

    wf::gles::bind_render_buffer(target);
    GL_CALL(glEnable(GL_BLEND));
    GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    

    self->animation_time += 0.016f;
    const float spacing_per_layer = 4.0f;

    for (int i = 20; i > 0; i--)
    {
        int spacing = static_cast<int>(i * spacing_per_layer);

        // --- FINAL CORRECTED LOGIC ---
        // We shift LEFT by the left margin.
        int local_start_x = 75 - spacing;
        // We shift UP by the top margin (since positive Y is up).
        int local_start_y = margins.top*2 - spacing;

        int layer_width = full_geo.width + 2 * spacing;
        int layer_height = full_geo.height + 2 * spacing;

        self->shader_program.use(wf::TEXTURE_TYPE_RGBA);
        self->shader_program.uniformMatrix4f("mvp", ortho);
        self->shader_program.uniform2f("resolution", (float)layer_width, (float)layer_height);
        self->shader_program.uniform1f("time", self->animation_time + i * 0.05f);

        float alpha = 0.05f + (1.0f - i / 20.0f) * 0.2f;
        glm::vec4 layer_color(1.0f, 1.0f, 1.0f, alpha);
        self->shader_program.uniform4f("color", layer_color);

        GLfloat x = local_start_x;
        GLfloat y = local_start_y;
        GLfloat w = layer_width;
        GLfloat h = layer_height;

        GLfloat vertexData[] = { x, y, x + w, y, x + w, y + h, x, y + h };
        GLfloat texData[] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };

        self->shader_program.attrib_pointer("position", 2, 0, vertexData);
        self->shader_program.attrib_pointer("texcoord", 2, 0, texData);
        GL_CALL(glDrawArrays(GL_TRIANGLE_FAN, 0, 4));
        self->shader_program.deactivate();
    }

    GL_CALL(glDisable(GL_BLEND));
  //  push_damage(self->get_bounding_box());

        
}
    };
};
}  // namespace wf

#endif  // BACKGROUND_SHADER_HPP