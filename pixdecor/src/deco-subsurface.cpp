#include "wayfire/geometry.hpp"
#include "wayfire/scene-input.hpp"
#include "wayfire/scene-operations.hpp"
#include "wayfire/scene-render.hpp"
#include "wayfire/scene.hpp"
#include "wayfire/signal-provider.hpp"
#include "wayfire/toplevel.hpp"
#include <memory>
#define GLM_FORCE_RADIANS
#include <glm/gtc/matrix_transform.hpp>

#include <linux/input-event-codes.h>

#include <wayfire/nonstd/wlroots.hpp>
#include <wayfire/output.hpp>
#include <wayfire/opengl.hpp>
#include <wayfire/core.hpp>
#include <wayfire/signal-definitions.hpp>
#include <wayfire/toplevel-view.hpp>
#include "deco-subsurface.hpp"
#include "deco-layout.hpp"
#include "deco-theme.hpp"
#include <wayfire/window-manager.hpp>
#include <wayfire/view-transform.hpp>
#include <wayfire/txn/transaction-manager.hpp>
#include <wayfire/scene-render.hpp>

#include <wayfire/plugins/common/cairo-util.hpp>


#include <cairo.h>
#include "shade.hpp"


namespace wf
{
namespace pixdecor
{

// Wayfire options
wf::option_wrapper_t<wf::color_t> effect_color{"pixdecor/effect_color"};
wf::option_wrapper_t<int> shadow_radius{"pixdecor/shadow_radius"};
wf::option_wrapper_t<std::string> titlebar_opt{"pixdecor/titlebar"};
wf::option_wrapper_t<int> csd_titlebar_height{"pixdecor/csd_titlebar_height"};
wf::option_wrapper_t<bool> enable_shade{"pixdecor/enable_shade"};
wf::option_wrapper_t<std::string> title_font{"pixdecor/title_font"};
wf::option_wrapper_t<std::string> overlay_engine{"pixdecor/overlay_engine"};
wf::option_wrapper_t<std::string> effect_type{"pixdecor/effect_type"};
wf::option_wrapper_t<bool> maximized_borders{"pixdecor/maximized_borders"};
wf::option_wrapper_t<bool> maximized_shadows{"pixdecor/maximized_shadows"};
wf::option_wrapper_t<int> title_text_align{"pixdecor/title_text_align"};
wf::option_wrapper_t<int> shader_extension{"pixdecor/shader_extension"};


class shader_background_node_t : public wf::scene::node_t
{
    std::weak_ptr<wf::toplevel_view_interface_t> _view;
    
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

// Simplex noise functions
vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec4 permute(vec4 x) {
    return mod289(((x*34.0)+1.0)*x);
}
vec4 taylorInvSqrt(vec4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}
float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
float normnoise(float noise) {
    return 0.5 * (noise + 1.0);
}
float clouds(vec2 uv, float t) {
    // Optimized: Reduced to 4 octaves, simpler motion
    uv += vec2(t * 0.03, t * 0.02);
    vec2 off1 = vec2(50.0, 33.0);
    vec2 off2 = vec2(0.0, 0.0);
    vec2 off3 = vec2(-300.0, 50.0);
    vec2 off4 = vec2(-100.0, 200.0);
    float scale1 = 3.0;
    float scale2 = 6.0;
    float scale3 = 12.0;
    float scale4 = 24.0;
    return normnoise(snoise(vec3((uv+off1)*scale1, t*0.5)) * 0.8 +
                     snoise(vec3((uv+off2)*scale2, t*0.4)) * 0.4 +
                     snoise(vec3((uv+off3)*scale3, t*0.1)) * 0.2 +
                     snoise(vec3((uv+off4)*scale4, t*0.7)) * 0.1);
}
// HSV to RGB conversion function
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Function to calculate the color at a given texture coordinate
vec4 get_color(vec2 uv) {
    vec2 center = vec2(0.5, 0.5);
    float t = time;
    const int NUM_LIGHTS = 4;
    vec3 finalColor = vec3(0.0);
    float cloudValue = clouds(uv, t);
    float invNum = 1.0 / float(NUM_LIGHTS);
    for(int i = 0; i < NUM_LIGHTS; i++) {
        float fi = float(i);
        float freq = 0.5 + fi * 0.1;
        vec2 light;
        if (i == 0) { // Bottom: left to right
            float xPos = 0.5 + 0.35 * sin(t * freq);
            float yPos = 0.10;
            light = vec2(xPos, yPos);
        } else if (i == 1) { // Top: left to right (opposite phase)
            float xPos = 0.5 + 0.35 * sin(t * freq + 3.14159);
            float yPos = 0.90;
            light = vec2(xPos, yPos);
        } else if (i == 2) { // Left: up and down
            float xPos = 0.10;
            float yPos = 0.5 + 0.35 * sin(t * freq);
            light = vec2(xPos, yPos);
        } else { // Right: up and down (opposite phase)
            float xPos = 0.90;
            float yPos = 0.5 + 0.35 * sin(t * freq + 3.14159);
            light = vec2(xPos, yPos);
        }
     
        vec3 lcolor = hsv2rgb(vec3(fi * 0.1 + fract(t * 0.1), 0.8, 1.0));
        vec2 dir;
        float aspect = 8.0;
        if (i == 0 || i == 1) {
            dir = vec2(1.0, 0.0);
        } else {
            dir = vec2(0.0, 1.0);
        }
        vec2 delta = uv - light;
        float along = dot(delta, dir);
        vec2 perp_delta = delta - along * dir;
        float perp = length(perp_delta);
        if (perp > 0.2) continue;
        float dist_along_scaled = along / aspect;
        float dist = length(vec2(perp, dist_along_scaled));
        if (dist > 0.6) continue;
        dist = max(dist, 0.001);
        float lightIntensity = 1.0 / (50.0 * dist * dist); // Quadratic falloff for quicker decay
        finalColor += lightIntensity * lcolor * invNum;
    }
    float alpha = length(finalColor) / sqrt(3.0);
    alpha = clamp(alpha * 1.2, 0.0, 1.0);
    
    vec2 from_center = abs(uv - 0.5);
    float max_dist = max(from_center.x, from_center.y);
    float vignette = 1.0 - smoothstep(0.4, 0.5, max_dist);
    alpha *= vignette;
    
    return vec4(finalColor * color.rgb, alpha * color.a);
}

void main() {
    vec2 uv = v_texcoord;
    vec4 original_color = get_color(uv);

    // Box blur
    float blur_amount = 0.002;
    vec4 blurred_color = vec4(0.0);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 offset = vec2(float(x), float(y)) * blur_amount;
            blurred_color += get_color(uv + offset);
        }
    }
    blurred_color /= 9.0;

    gl_FragColor = blurred_color;
}
)";

    void init_shader()
    {
        if (shader_initialized) return;
        
        shader_program.set_simple(
            OpenGL::compile_program(vertex_shader, fragment_shader));
        shader_initialized = true;
    }

public:
    wf::dimensions_t size;
    int current_thickness = 0;
    int current_titlebar = 0;
    wf::output_t *output;  // NEW: Store output reference for screen size

    shader_background_node_t(wayfire_toplevel_view view) : node_t(false)
    {
        this->_view = view->weak_from_this();
        output = view->get_output();  // NEW: Capture output
    }

    ~shader_background_node_t()
    {
        shader_program.free_resources();
    }

    wf::point_t get_offset()
    {
        auto view = _view.lock();
        if (view && view->pending_tiled_edges() && !maximized_borders && !maximized_shadows)
        {
            return {0, -current_titlebar};
        }
        return {-current_thickness, -current_titlebar};
    }
/*
    wf::geometry_t get_bounding_box() override
{
    // MODIFIED: Use output->handle->width and output->handle->height
    return wf::geometry_t{
        -static_cast<int>(output->handle->width), -static_cast<int>(output->handle->height)*5,  // Start at output origin (0,0)
        static_cast<int>(output->handle->width)*2, static_cast<int>(output->handle->height)*10
    };
}*/


wf::geometry_t get_bounding_box() override
    {
        auto offset = get_offset();
        int extension = int(shader_extension);
        return wf::geometry_t{
            offset.x - extension*4,
            offset.y - extension*4,
            size.width + 2 * extension*4,
            size.height + 2 * extension*4
        };
    }

 


    void render_shader(const wf::scene::render_instruction_t& data, wf::point_t origin)
    {
        auto view = _view.lock();
        if (!view) return;
        init_shader();
        int extension = int(shader_extension);  // No longer needed

        const auto& target = data.target;
        int fb_width = target.geometry.width;
        int fb_height = target.geometry.height;
        auto ortho = glm::ortho(0.0f, (float)fb_width, (float)fb_height, 0.0f);
        wf::gles::bind_render_buffer(target);
        GL_CALL(glEnable(GL_BLEND));
        GL_CALL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

        animation_time += 0.016f;
        const float spacing_per_layer = 40.0f;  // Kept for potential multi-layer use

        {
            // MODIFIED: Use full screen size instead of window + spacing/extension
            int layer_width = fb_width;   // Or output->width if target != output
            int layer_height = fb_height; // Or output->height if target != output
            shader_program.use(wf::TEXTURE_TYPE_RGBA);
            shader_program.uniformMatrix4f("mvp", ortho);
            // MODIFIED: Resolution is now full screen for consistent scaling
            shader_program.uniform2f("resolution", (float)layer_width, (float)layer_height);
            shader_program.uniform1f("time", animation_time + 0 * 0.05f);
            float alpha = 0.05f + (1.0f / 20.0f) * 0.2f;
            glm::vec4 layer_color(1.0f, 1.0f, 1.0f, 1.0);
            shader_program.uniform4f("color", layer_color);
            // MODIFIED: Quad covers full render target (screen)
            GLfloat x = 0.0f;  // REMOVED: local_start_x/origin calculations
            GLfloat y = 0.0f;
            GLfloat w = (GLfloat)layer_width;
            GLfloat h = (GLfloat)layer_height;
            GLfloat vertexData[] = { x, y, x + w, y, x + w, y + h, x, y + h };
            GLfloat texData[] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
            shader_program.attrib_pointer("position", 2, 0, vertexData);
            shader_program.attrib_pointer("texcoord", 2, 0, texData);
            GL_CALL(glDrawArrays(GL_TRIANGLE_FAN, 0, 4));
            shader_program.deactivate();
        }
        GL_CALL(glDisable(GL_BLEND));

        // Schedule next frame damage to keep animation running
        // Also damage the view's surface to prevent artifacts
        wf::scene::damage_node(shared_from_this(), get_bounding_box());
        if (view)
        {
            view->damage();
        }
    }


    class shader_render_instance_t : public wf::scene::render_instance_t
    {
        shader_background_node_t *self;
        wf::scene::damage_callback push_damage;
        wf::signal::connection_t<wf::scene::node_damage_signal> on_node_damage =
            [=] (wf::scene::node_damage_signal *data) { push_damage(data->region); };

    public:
        shader_render_instance_t(shader_background_node_t *self, wf::scene::damage_callback push_damage)
            : self(self), push_damage(push_damage)
        {
            self->connect(&on_node_damage);
        }

        void schedule_instructions(std::vector<wf::scene::render_instruction_t>& instructions,
            const wf::render_target_t& target, wf::region_t& damage) override
        {
            auto bbox = self->get_bounding_box();
            wf::region_t our_region{bbox};
            wf::region_t our_damage = damage & our_region;
            
            if (!our_damage.empty())
            {
                instructions.push_back(wf::scene::render_instruction_t{
                    .instance = this,
                    .target   = target,
                    .damage   = std::move(our_damage),
                });
            }
        }

        void render(const wf::scene::render_instruction_t& data) override
        {
            auto offset = self->get_offset();
            self->render_shader(data, offset);
        }
    };

    void gen_render_instances(std::vector<wf::scene::render_instance_uptr>& instances,
        wf::scene::damage_callback push_damage, wf::output_t *output = nullptr) override
    {
        instances.push_back(std::make_unique<shader_render_instance_t>(this, push_damage));
    }

    void update_size(wf::dimensions_t dims, int thickness, int titlebar)
    {
        auto view = _view.lock();
        
        // Damage the old bounding box first
        wf::scene::damage_node(shared_from_this(), get_bounding_box());
        if (view)
        {
            view->damage();
        }
        
        // Update size parameters
        size = dims;
        current_thickness = thickness;
        current_titlebar = titlebar;
        
        // Damage the new bounding box
        wf::scene::damage_node(shared_from_this(), get_bounding_box());
        if (view)
        {
            view->damage();
        }
    }
};

class simple_decoration_node_t : public wf::scene::node_t, public wf::pointer_interaction_t,
    public wf::touch_interaction_t
{
    std::weak_ptr<wf::toplevel_view_interface_t> _view;
    
   


 
    wf::signal::connection_t<wf::view_title_changed_signal> title_set =
        [=] (wf::view_title_changed_signal *ev)
    {
        // Title change just requires redrawing the decoration.
        wf::scene::damage_node(shared_from_this(), get_bounding_box());
    };

    void update_title(int width, int height, int t_width, int border, int buttons_width, double scale)
    {
        auto view = _view.lock();
        if (!view)
        {
            return;
        }

        int target_width  = width * scale;
        int target_height = height * scale;

        // Check if the title texture needs to be regenerated.
        if ((int(title_text_align) != title_texture.title_text_align) ||
            (view->get_title() != title_texture.current_text) ||
            (target_width != title_texture.tex.get_size().width) ||
            (std::string(title_font) != title_texture.title_font_string) ||
            (target_height != title_texture.tex.get_size().height) ||
            (view->activated != title_texture.rendered_for_activated_state))
        {
            auto surface = theme.render_text(view->get_title(),
                target_width, target_height, t_width, border, buttons_width, view->activated);
            title_texture.tex = owned_texture_t{surface};
            cairo_surface_destroy(surface);
            
            // Update cached state
            title_texture.title_font_string = title_font;
            title_texture.current_text     = view->get_title();
            title_texture.title_text_align = int(title_text_align);
            title_texture.rendered_for_activated_state = view->activated;
        }
    }

    struct
    {
        wf::owned_texture_t tex;
        std::string current_text = "";
        bool rendered_for_activated_state = false;
        int title_text_align = 0; // Initialize to a default
        std::string title_font_string = "";
    } title_texture;

  public:
    pixdecor_theme_t theme;
    pixdecor_layout_t layout;
    wf::region_t cached_region;
    wf::dimensions_t size;
    int current_thickness;
    int current_titlebar;
    wf::pointf_t current_cursor_position;


simple_decoration_node_t(wayfire_toplevel_view view) :
    node_t(false),
    theme{},
    layout{theme, [=] (wlr_box box) { wf::scene::damage_node(shared_from_this(), box + get_offset()); }}
{
    this->_view = view->weak_from_this();
    view->connect(&title_set);
    current_cursor_position.x = current_cursor_position.y = FLT_MIN;
    
}

 ~simple_decoration_node_t()
{
    remove_shade_transformers();
}

    wf::point_t get_offset()
    {
        auto view = _view.lock();
        if (view && view->pending_tiled_edges() && !maximized_borders && !maximized_shadows)
        {
            return {0, -current_titlebar};
        }

        return {-current_thickness, -current_titlebar};
    }

    void render_title(const wf::scene::render_instruction_t& data,
        const wf::geometry_t& geometry, int t_width, int border, int buttons_width)
    {
        update_title(geometry.width, geometry.height, t_width, border, buttons_width, data.target.scale);
        OpenGL::render_texture(wf::gles_texture_t{title_texture.tex.get_texture()}, data.target, geometry,
            glm::vec4(1.0f), OpenGL::RENDER_FLAG_CACHED);

        // This seems to be an optimization for partial damage, it's fine as is.
        data.pass->custom_gles_subpass(data.target, [&]
        {
            for (auto& box : data.damage)
            {
                wf::gles::render_target_logic_scissor(data.target, wlr_box_from_pixman_box(box));
                OpenGL::draw_cached();
            }
        });

        OpenGL::clear_cached();
    }

    // Helper to determine if the titlebar should be rendered based on current state and options
    bool should_render_titlebar(bool maximized)
    {
        const std::string titlebar_setting = titlebar_opt;
        if (titlebar_setting == "always") return true;
        if (titlebar_setting == "never") return false;
        if ((titlebar_setting == "maximized") && !maximized) return false;
        if ((titlebar_setting == "windowed") && maximized) return false;
        return true;
    }

    void render_region(const wf::scene::render_instruction_t& data, wf::point_t origin)
    {
        int border = theme.get_border_size();
        wlr_box geometry{origin.x, origin.y, size.width, size.height};

        bool activated = false;
        bool maximized = false;
        if (auto view = _view.lock())
        {
            activated = view->activated;
            maximized = view->pending_tiled_edges();
        }

        theme.render_background(data, geometry, activated, current_cursor_position);

        if (!should_render_titlebar(maximized))
        {
            return;
        }

        auto renderables = layout.get_renderable_areas();
        int buttons_width = 0;
        for (auto item : renderables)
        {
            if (item->get_type() != DECORATION_AREA_TITLE)
            {
                buttons_width += item->get_geometry().width;
            }
        }

        const bool use_rounded_corners = (std::string(overlay_engine) == "rounded_corners" && (!maximized || maximized_shadows));
        int title_border = border + (use_rounded_corners ? int(shadow_radius) * 2 : 0);

        for (auto item : renderables)
        {
            if (item->get_type() == DECORATION_AREA_TITLE)
            {
                render_title(data,
                    item->get_geometry() + origin, size.width - border * 2, title_border, buttons_width);
            } else // button
            {
                item->as_button().render(data,
                    item->get_geometry() + origin);
            }
        }
    }

    std::optional<wf::scene::input_node_t> find_node_at(const wf::pointf_t& at) override
    {
        auto view = _view.lock();
        if (!view)
        {
            return {};
        }

        bool maximized = view->pending_tiled_edges();
        int border = theme.get_border_size();
        int r = (std::string(overlay_engine) == "rounded_corners" && (!maximized || (maximized && maximized_shadows))) 
            ? int(shadow_radius) * 2 
            : 0;
        r -= MIN_RESIZE_HANDLE_SIZE - std::min(border, MIN_RESIZE_HANDLE_SIZE);

        wf::pointf_t local = at - wf::pointf_t{get_offset()};
        
        wf::geometry_t g = view->get_geometry();
        g.x = g.y = 0;
        g = wf::expand_geometry_by_margins(g, wf::decoration_margins_t{-r, -r, -r, -r});
        wf::region_t deco_region{g};

        g = wf::expand_geometry_by_margins(g, wf::decoration_margins_t{-border, -border, -border, -theme.get_title_height() - border});
        wf::region_t view_region{g};
        deco_region ^= view_region;

        if (deco_region.contains_pointf(local))
        {
            return wf::scene::input_node_t{
                .node = this,
                .local_coords = local,
            };
        }
        
        return {};
    }

    pointer_interaction_t& pointer_interaction() override { return *this; }
    touch_interaction_t& touch_interaction() override { return *this; }

    class decoration_render_instance_t : public wf::scene::render_instance_t
    {
        simple_decoration_node_t *self;
        wf::scene::damage_callback push_damage;
        wf::signal::connection_t<wf::scene::node_damage_signal> on_surface_damage =
            [=] (wf::scene::node_damage_signal *data) { push_damage(data->region); };

      public:
        decoration_render_instance_t(simple_decoration_node_t *self, wf::scene::damage_callback push_damage)
            : self(self), push_damage(push_damage)
        {
            self->connect(&on_surface_damage);
        }

void schedule_instructions(std::vector<wf::scene::render_instruction_t>& instructions,
    const wf::render_target_t& target, wf::region_t& damage) override
{
    auto our_region = self->cached_region + self->get_offset();
    wf::region_t our_damage = damage & our_region;
    
    if (!our_damage.empty())
    {
        instructions.push_back(wf::scene::render_instruction_t{
            .instance = this,
            .target   = target,
            .damage   = std::move(our_damage),
        });
    }
}

void render(const wf::scene::render_instruction_t& data) override
{
    auto offset = self->get_offset();
    wlr_box rectangle{offset.x, offset.y, self->size.width, self->size.height};
    bool activated = false;
    bool maximized = false;
    if (auto view = self->_view.lock())
    {
        activated = view->activated;
        maximized = maximized_shadows ? false : view->pending_tiled_edges();
    }



    if ((std::string(effect_type) != "none") || (std::string(overlay_engine) != "none"))
    {
        self->theme.smoke.step_effect(data, rectangle, std::string(effect_type) == "ink",
            self->current_cursor_position, self->theme.get_decor_color(activated), effect_color,
            self->theme.get_title_height(), self->theme.get_border_size(),
            (std::string(overlay_engine) == "rounded_corners" && !maximized) ? shadow_radius : 0);
    }

    self->render_region(data, offset);
}
    };



    void gen_render_instances(std::vector<wf::scene::render_instance_uptr>& instances,
        wf::scene::damage_callback push_damage, wf::output_t *output = nullptr) override
    {
        instances.push_back(std::make_unique<decoration_render_instance_t>(this, push_damage));
    }

  wf::geometry_t get_bounding_box() override
{
    auto offset = get_offset();
    int extension = int(shader_extension);
    return wf::geometry_t{
        offset.x - extension,
        offset.y - extension,
        size.width + 2 * extension,
        size.height + 2 * extension
    };
}

    void handle_pointer_enter(wf::pointf_t point) override
    {
        point -= wf::pointf_t{get_offset()};
        layout.handle_motion(point.x, point.y);
    }

    void handle_pointer_leave() override
    {
        layout.handle_focus_lost();
        current_cursor_position.x = current_cursor_position.y = FLT_MIN;
    }

    void handle_pointer_motion(wf::pointf_t to, uint32_t) override
    {
        to -= wf::pointf_t{get_offset()};
        handle_action(layout.handle_motion(to.x, to.y));
        current_cursor_position = to;
    }

    void handle_pointer_button(const wlr_pointer_button_event& ev) override
    {
        if (ev.button != BTN_LEFT) return;
        handle_action(layout.handle_press_event(ev.state == WL_POINTER_BUTTON_STATE_PRESSED));
    }

    void handle_pointer_axis(const wlr_pointer_axis_event& ev) override
    {
        if (ev.orientation == WL_POINTER_AXIS_VERTICAL_SCROLL)
        {
            handle_action(layout.handle_axis_event(ev.delta));
        }
    }

    void pop_transformer(wayfire_view view)
    {
        if (view->get_transformed_node()->get_transformer(shade_transformer_name))
        {
            view->get_transformed_node()->rem_transformer(shade_transformer_name);
        }
    }

    void remove_shade_transformers()
    {
        for (auto& view : wf::get_core().get_all_views())
        {
            pop_transformer(view);
        }
    }

    std::shared_ptr<pixdecor_shade> ensure_transformer(wayfire_view view, int titlebar_height)
    {
        auto tmgr = view->get_transformed_node();
        if (auto tr = tmgr->get_transformer<pixdecor_shade>(shade_transformer_name))
        {
            return tr;
        }
        auto node = std::make_shared<pixdecor_shade>(view, titlebar_height);
        tmgr->add_transformer(node, wf::TRANSFORMER_2D, shade_transformer_name);
        return tmgr->get_transformer<pixdecor_shade>(shade_transformer_name);
    }

    void init_shade(wayfire_view view, bool shade, int titlebar_height)
    {
        if (!bool(enable_shade)) return;

        if (shade)
        {
            if (view && view->is_mapped())
            {
                auto tr = ensure_transformer(view, titlebar_height);
                tr->set_titlebar_height(titlebar_height);
                tr->init_animation(shade);
            }
        } else
        {
            if (auto tr = view->get_transformed_node()->get_transformer<pixdecor_shade>(shade_transformer_name))
            {
                tr->set_titlebar_height(titlebar_height);
                tr->init_animation(shade);
            }
        }
    }

    void handle_action(pixdecor_layout_t::action_response_t action)
    {
        auto view = _view.lock();
        if (!view) return;

        switch (action.action)
        {
            case DECORATION_ACTION_MOVE:
                return wf::get_core().default_wm->move_request(view);
            case DECORATION_ACTION_RESIZE:
                return wf::get_core().default_wm->resize_request(view, action.edges);
            case DECORATION_ACTION_CLOSE:
                return view->close();
            case DECORATION_ACTION_TOGGLE_MAXIMIZE:
                if (view->pending_tiled_edges())
                {
                    wf::get_core().default_wm->tile_request(view, 0);
                } else
                {
                    wf::get_core().default_wm->tile_request(view, wf::TILED_EDGES_ALL);
                }
                break;
            case DECORATION_ACTION_SHADE:
                init_shade(view, true, current_titlebar);
                break;
            case DECORATION_ACTION_UNSHADE:
                init_shade(view, false, current_titlebar);
                break;
            case DECORATION_ACTION_MINIMIZE:
                wf::get_core().default_wm->minimize_request(view, true);
                break;
            default:
                break;
        }
    }

    void handle_touch_down(uint32_t time_ms, int finger_id, wf::pointf_t position) override
    {
        handle_touch_motion(time_ms, finger_id, position);
        handle_action(layout.handle_press_event());
    }

    void handle_touch_up(uint32_t time_ms, int finger_id, wf::pointf_t lift_off_position) override
    {
        handle_action(layout.handle_press_event(false));
        layout.handle_focus_lost();
    }

    void handle_touch_motion(uint32_t time_ms, int finger_id, wf::pointf_t position) override
    {
        position -= wf::pointf_t{get_offset()};
        layout.handle_motion(position.x, position.y);
        current_cursor_position = position;
    }

    void recreate_frame()
    {
        update_decoration_size();
        if (auto view = _view.lock())
        {
            auto size = wf::dimensions(view->get_pending_geometry());
            layout.resize(size.width, size.height);
            wf::get_core().tx_manager->schedule_object(view->toplevel());
        }
    }

void resize(wf::dimensions_t dims)
{
    auto view = _view.lock();
    if (!view) return;

    // Damage EVERYTHING before changing
    view->damage();
    wf::scene::damage_node(shared_from_this(), get_bounding_box());

    theme.set_maximize(view->pending_tiled_edges());
    layout.set_maximize(maximized_shadows ? 0 : view->pending_tiled_edges());
    
    size = dims;
    layout.resize(size.width, size.height);
    if (!view->toplevel()->current().fullscreen)
    {
        this->cached_region = layout.calculate_region();
    }
    
    // Damage EVERYTHING after changing
    wf::scene::damage_node(shared_from_this(), get_bounding_box());
    view->damage();
}

    void update_decoration_size()
    {
        auto view = _view.lock();
        if (!view) return;

        bool fullscreen = view->toplevel()->pending().fullscreen;
        bool maximized  = view->toplevel()->pending().tiled_edges;

        if (fullscreen)
        {
            current_thickness = 0;
            current_titlebar  = 0;
            this->cached_region.clear();
        } else
        {
            int shadow_thickness = std::string(overlay_engine) == "rounded_corners" &&
                (!maximized || (maximized && maximized_shadows)) ? int(shadow_radius) * 2 : 0;

            current_thickness = theme.get_border_size() + shadow_thickness;
            
            if (should_render_titlebar(maximized) || (maximized && (maximized_borders || maximized_shadows)))
            {
                 current_titlebar = theme.get_title_height() + current_thickness;
            } else
            {
                current_titlebar = 0;
            }
            this->cached_region = layout.calculate_region();
        }

        if (auto tr = view->get_transformed_node()->get_transformer<pixdecor_shade>(shade_transformer_name))
        {
            tr->set_titlebar_height(current_titlebar);
        }

        wf::scene::damage_node(shared_from_this(), get_bounding_box());
    }
};

simple_decorator_t::simple_decorator_t(wayfire_toplevel_view view)
{
    this->view = view;
    this->shadow_thickness = 0;

    // Create shader background node FIRST (renders behind)
    shader_node = std::make_shared<shader_background_node_t>(view);
    wf::scene::add_back(view->get_surface_root_node(), shader_node);
    
    // Create decoration node SECOND (renders on top)
    deco = std::make_shared<simple_decoration_node_t>(view);
    wf::scene::add_front(view->get_surface_root_node(), deco);

    deco->update_decoration_size();
    deco->resize(wf::dimensions(view->get_pending_geometry()));
    
    // Initialize shader node size
    shader_node->update_size(
        wf::dimensions(view->get_pending_geometry()),
        deco->current_thickness,
        deco->current_titlebar
    );

  auto handle_state_change = [this] ()
{
    deco->update_decoration_size();  // This updates current_thickness and current_titlebar
    deco->resize(wf::dimensions(this->view->get_geometry()));
    
    // Update shader node with the NEW decoration parameters
    shader_node->update_size(
        wf::dimensions(this->view->get_geometry()),
        deco->current_thickness,
        deco->current_titlebar
    );
    
    wf::get_core().tx_manager->schedule_object(this->view->toplevel());
};

    on_view_activated = [this] (auto)
    {
        wf::scene::damage_node(deco, deco->get_bounding_box());
        wf::scene::damage_node(shader_node, shader_node->get_bounding_box());
    };

on_view_geometry_changed = [this] (auto)
{
    // Force complete redraw
    this->view->damage();
    
    wf::scene::damage_node(shader_node, shader_node->get_bounding_box());
    wf::scene::damage_node(deco, deco->get_bounding_box());
    
    deco->update_decoration_size();
    deco->resize(wf::dimensions(this->view->get_geometry()));
    
    shader_node->update_size(
        wf::dimensions(this->view->get_geometry()),
        deco->current_thickness,
        deco->current_titlebar
    );
    
    wf::get_core().tx_manager->schedule_object(this->view->toplevel());
    
    // Force complete redraw again
    this->view->damage();
    wf::scene::damage_node(shader_node, shader_node->get_bounding_box());
    wf::scene::damage_node(deco, deco->get_bounding_box());
};

    on_view_tiled = [=] (auto) { handle_state_change(); };
    on_view_fullscreen = [=] (auto) { handle_state_change(); };

    view->connect(&on_view_activated);
    view->connect(&on_view_geometry_changed);
    view->connect(&on_view_fullscreen);
    view->connect(&on_view_tiled);
}

simple_decorator_t::~simple_decorator_t()
{
    wf::scene::remove_child(shader_node);
    wf::scene::remove_child(deco);
}

int simple_decorator_t::get_titlebar_height()
{
    return deco->current_titlebar;
}

void simple_decorator_t::recreate_frame()
{
    deco->recreate_frame();
}

void simple_decorator_t::update_decoration_size()
{
    deco->update_decoration_size();
}

void simple_decorator_t::update_colors()
{
    deco->theme.update_colors();
}

void simple_decorator_t::effect_updated()
{
    deco->theme.smoke.effect_updated();
}

wf::decoration_margins_t simple_decorator_t::get_margins(const wf::toplevel_state_t& state)
{
    if (state.fullscreen)
    {
        return {0, 0, 0, 0};
    }

    bool maximized = state.tiled_edges;
    deco->theme.set_maximize(maximized);

    this->shadow_thickness = (std::string(overlay_engine) == "rounded_corners" &&
        (!state.tiled_edges || (state.tiled_edges && maximized_shadows))) ? int(shadow_radius) * 2 : 0;

    int thickness = deco->theme.get_border_size() + this->shadow_thickness;
    int titlebar  = deco->theme.get_title_height() + thickness;

    const std::string titlebar_setting = titlebar_opt;
    bool should_hide_titlebar = (titlebar_setting == "never") ||
                                (titlebar_setting == "maximized" && !maximized) ||
                                (titlebar_setting == "windowed" && maximized);

    if (maximized && should_hide_titlebar && !maximized_borders && !maximized_shadows)
    {
        titlebar = 0;
    }

    if (state.tiled_edges && !maximized_borders)
    {
        if (maximized_shadows)
        {
            if (should_hide_titlebar)
            {
                titlebar = thickness;
            }
        } else
        {
            thickness = 0;
        }
    }

    double shade_progress = 0.0;
    if (auto tr = view->get_transformed_node()->get_transformer<pixdecor_shade>(shade_transformer_name))
    {
        tr->set_titlebar_height(titlebar);
        shade_progress = tr->progression.shade;
    }

    int bottom_shadow_margin = shadow_thickness + int((view->get_geometry().height - shadow_thickness - titlebar) * shade_progress);

    if (view->has_data(custom_data_name))
    {
        view->get_data<wf_shadow_margin_t>(custom_data_name)->set_margins(
            {shadow_thickness, shadow_thickness, shadow_thickness, bottom_shadow_margin});
    } else
    {
        view->store_data(std::make_unique<wf_shadow_margin_t>(), custom_data_name);
        view->get_data<wf_shadow_margin_t>(custom_data_name)->set_margins(
            {shadow_thickness, shadow_thickness, shadow_thickness, bottom_shadow_margin});
    }

    return wf::decoration_margins_t{
        .left   = thickness,
        .right  = thickness,
        .bottom = thickness,
        .top    = titlebar,
    };
}

void simple_decorator_t::update_animation()
{
    auto margins = get_margins(view->toplevel()->current());
    auto bbox    = deco->get_bounding_box();

    wf::region_t region;
    region |= wlr_box{bbox.x, bbox.y, bbox.width, margins.top};
    region |= wlr_box{bbox.x, bbox.y, margins.left, bbox.height};
    region |= wlr_box{bbox.x, bbox.y + bbox.height - margins.bottom, bbox.width, margins.bottom};
    region |= wlr_box{bbox.x + bbox.width - margins.right, bbox.y, margins.right, bbox.height};
    wf::scene::damage_node(deco, region);
    
    // Also damage the shader node during animations
    wf::scene::damage_node(shader_node, shader_node->get_bounding_box());
}

} // namespace pixdecor
} // namespace wf