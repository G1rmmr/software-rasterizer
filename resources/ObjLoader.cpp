#include "ObjLoader.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

namespace {
    struct Vec3 {
        double X = 0, Y = 0, Z = 0;
        Vec3 operator+(Vec3 b) const { return {X + b.X, Y + b.Y, Z + b.Z}; }
        Vec3 operator-(Vec3 b) const { return {X - b.X, Y - b.Y, Z - b.Z}; }
        Vec3 operator*(double s) const { return {X * s, Y * s, Z * s}; }
        double Dot(Vec3 b) const { return X * b.X + Y * b.Y + Z * b.Z; }
        Vec3 Cross(Vec3 b) const { return {Y * b.Z - Z * b.Y, Z * b.X - X * b.Z, X * b.Y - Y * b.X}; }
        double Length() const { return std::hypot(X, Y, Z); }
        Vec3 Unit() const {
            const double length = Length();
            return length > 0 ? *this * (1.0 / length) : Vec3{};
        }
        math::Vector Direction(float w = 0.f) const {
            return {static_cast<float>(X), static_cast<float>(Y), static_cast<float>(Z), w};
        }
    };

    Vec3 XYZ(const math::Vector& v) {
        return {v.X, v.Y, v.Z};
    }
    constexpr std::size_t Missing = std::numeric_limits<std::size_t>::max();
    using NormalKey = std::pair<std::size_t, std::int64_t>;
    using VertexKey = std::tuple<std::size_t, std::size_t, std::size_t, std::int64_t>;

    class Parser {
    public:
        explicit Parser(std::string source) : source(std::move(source)) {}

        resources::ObjData Read(std::istream& input) {
            std::string line;
            while(std::getline(input, line)) {
                ++lineNumber;
                if(const auto comment = line.find('#'); comment != std::string::npos) line.resize(comment);
                std::istringstream stream(line);
                std::string kind;
                if(!(stream >> kind)) continue;
                if(kind == "v") {
                    auto values = Numbers(stream);
                    if(values.size() != 3 && values.size() != 4) Fail("vertex requires x y z and optional w");
                    const double w = values.size() == 4 ? values[3] : 1.0;
                    if(w == 0) Fail("vertex homogeneous w must not be zero");
                    Vec3 point{values[0] / w, values[1] / w, values[2] / w};
                    if(!FitsFloat(point.X) || !FitsFloat(point.Y) || !FitsFloat(point.Z))
                        Fail("vertex exceeds float range");
                    positions.push_back(point.Direction(1.f));
                }
                else if(kind == "vt") {
                    auto values = Numbers(stream);
                    if(values.empty() || values.size() > 3) Fail("texture coordinate requires one to three values");
                    uvs.emplace_back(static_cast<float>(values[0]),
                                     values.size() > 1 ? static_cast<float>(values[1]) : 0.f, 0.f, 0.f);
                }
                else if(kind == "vn") {
                    auto values = Numbers(stream);
                    if(values.size() != 3) Fail("normal requires x y z");
                    Vec3 normal{values[0], values[1], values[2]};
                    if(normal.Length() == 0) Fail("normal must have nonzero length");
                    normals.push_back(normal.Unit().Direction());
                }
                else if(kind == "s") {
                    std::string token, extra;
                    if(!(stream >> token) || (stream >> extra)) Fail("invalid smoothing group");
                    if(token == "off" || token == "0")
                        smoothingGroup = 0;
                    else if(token == "on")
                        smoothingGroup = 1;
                    else {
                        smoothingGroup = Integer(token);
                        if(smoothingGroup <= 0) Fail("smoothing group must be positive or off");
                    }
                }
                else if(kind == "f")
                    ReadFace(stream);
                // Object/group/material names describe asset organization, not geometry.
                // ModelFactory supplies the material binding explicitly.
            }
            if(input.bad()) Fail("I/O error while reading OBJ");
            if(result.Indices.empty()) Fail("OBJ contains no faces");
            FinishAttributes();
            return std::move(result);
        }

    private:
        std::string source;
        std::size_t lineNumber = 0;
        std::int64_t faceNumber = 0;
        std::int64_t smoothingGroup = 0;
        std::vector<math::Vector> positions, uvs, normals;
        resources::ObjData result;
        std::map<VertexKey, std::uint32_t> vertexCache;
        std::map<NormalKey, Vec3> normalSums;
        std::vector<NormalKey> generatedNormalKeys;
        std::vector<bool> needsNormal;

        [[noreturn]] void Fail(const std::string& detail) const {
            throw std::runtime_error(source + ":" + std::to_string(lineNumber) + ": " + detail);
        }

        static bool FitsFloat(double v) { return std::isfinite(v) && std::abs(v) <= std::numeric_limits<float>::max(); }

        std::vector<double> Numbers(std::istringstream& stream) const {
            std::vector<double> values;
            double value;
            while(stream >> value) {
                if(!FitsFloat(value)) Fail("coordinate must be finite and fit a float");
                values.push_back(value);
            }
            if(!stream.eof()) Fail("invalid numeric coordinate");
            return values;
        }

        std::int64_t Integer(std::string_view text) const {
            if(!text.empty() && text.front() == '+') text.remove_prefix(1);
            std::int64_t value = 0;
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
            if(error != std::errc{} || end != text.data() + text.size()) Fail("invalid OBJ index");
            return value;
        }

        std::size_t Index(std::string_view text, std::size_t count) const {
            const std::int64_t index = Integer(text);
            if(index == 0 || count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
                Fail("OBJ index is out of range");
            if(index > 0) {
                if(static_cast<std::uint64_t>(index) > count) Fail("OBJ index is out of range");
                return static_cast<std::size_t>(index - 1);
            }
            if(index < -static_cast<std::int64_t>(count)) Fail("relative OBJ index is out of range");
            return static_cast<std::size_t>(static_cast<std::int64_t>(count) + index);
        }

        std::uint32_t FaceVertex(const std::string& token) {
            std::array<std::string_view, 3> pieces{};
            std::size_t pieceCount = 0, begin = 0;
            while(true) {
                if(pieceCount == pieces.size()) Fail("too many fields in a face vertex");
                const auto slash = token.find('/', begin);
                pieces[pieceCount++] =
                    std::string_view(token).substr(begin, slash == std::string::npos ? slash : slash - begin);
                if(slash == std::string::npos) break;
                begin = slash + 1;
            }
            if(pieces[0].empty() || (pieceCount > 1 && pieces[pieceCount - 1].empty())) Fail("incomplete face vertex");
            const auto p = Index(pieces[0], positions.size());
            const auto uv = pieceCount > 1 && !pieces[1].empty() ? Index(pieces[1], uvs.size()) : Missing;
            const auto normal = pieceCount == 3 ? Index(pieces[2], normals.size()) : Missing;
            const auto group = normal == Missing ? (smoothingGroup == 0 ? -faceNumber : smoothingGroup) : 0;
            const VertexKey key{p, uv, normal, group};
            if(auto found = vertexCache.find(key); found != vertexCache.end()) return found->second;
            if(result.Vertices.size() >= std::numeric_limits<std::uint32_t>::max()) Fail("too many vertices");
            const auto index = static_cast<std::uint32_t>(result.Vertices.size());
            graphics::Vertex vertex;
            vertex.Pos = positions[p];
            vertex.Normal = normal == Missing ? math::Vector{} : normals[normal];
            vertex.UV = uv == Missing ? math::Vector{} : uvs[uv];
            result.Vertices.push_back(vertex);
            generatedNormalKeys.emplace_back(p, group);
            needsNormal.push_back(normal == Missing);
            vertexCache.emplace(key, index);
            return index;
        }

        void ReadFace(std::istringstream& stream) {
            ++faceNumber;
            std::vector<std::uint32_t> face;
            std::string token;
            while(stream >> token) face.push_back(FaceVertex(token));
            if(face.size() < 3) Fail("face requires at least three vertices");
            for(std::size_t i = 1; i + 1 < face.size(); ++i) {
                const auto a = face[0], b = face[i], c = face[i + 1];
                const auto edge1 = XYZ(result.Vertices[b].Pos) - XYZ(result.Vertices[a].Pos);
                const auto edge2 = XYZ(result.Vertices[c].Pos) - XYZ(result.Vertices[a].Pos);
                const auto normal = edge1.Cross(edge2);
                if(normal.Length() == 0) Fail("face contains a degenerate triangle");
                result.Indices.insert(result.Indices.end(), {a, b, c});
                for(auto index : {a, b, c}) {
                    if(needsNormal[index]) {
                        auto& sum = normalSums[generatedNormalKeys[index]];
                        sum = sum + normal;
                    }
                }
            }
        }

        void FinishAttributes() {
            for(std::size_t i = 0; i < result.Vertices.size(); ++i) {
                if(needsNormal[i]) {
                    const Vec3 normal = normalSums.at(generatedNormalKeys[i]);
                    if(normal.Length() == 0) Fail("faces cancel to an undefined vertex normal");
                    result.Vertices[i].Normal = normal.Unit().Direction();
                }
            }
            std::vector<Vec3> tangentSums(result.Vertices.size()), bitangentSums(result.Vertices.size());
            for(std::size_t i = 0; i < result.Indices.size(); i += 3) {
                const auto a = result.Indices[i], b = result.Indices[i + 1], c = result.Indices[i + 2];
                const auto& v0 = result.Vertices[a];
                const auto& v1 = result.Vertices[b];
                const auto& v2 = result.Vertices[c];
                const auto e1 = XYZ(v1.Pos) - XYZ(v0.Pos), e2 = XYZ(v2.Pos) - XYZ(v0.Pos);
                const double du1 = static_cast<double>(v1.UV.X) - v0.UV.X;
                const double dv1 = static_cast<double>(v1.UV.Y) - v0.UV.Y;
                const double du2 = static_cast<double>(v2.UV.X) - v0.UV.X;
                const double dv2 = static_cast<double>(v2.UV.Y) - v0.UV.Y;
                const double determinant = du1 * dv2 - du2 * dv1;
                if(std::abs(determinant) <= 1e-10 * (std::abs(du1 * dv2) + std::abs(du2 * dv1))) continue;
                const auto tangent = (e1 * dv2 - e2 * dv1) * (1.0 / determinant);
                const auto bitangent = (e2 * du1 - e1 * du2) * (1.0 / determinant);
                for(auto index : {a, b, c}) {
                    tangentSums[index] = tangentSums[index] + tangent;
                    bitangentSums[index] = bitangentSums[index] + bitangent;
                }
            }
            for(std::size_t i = 0; i < result.Vertices.size(); ++i) {
                const auto normal = XYZ(result.Vertices[i].Normal).Unit();
                auto tangent = tangentSums[i] - normal * normal.Dot(tangentSums[i]);
                if(tangent.Length() == 0) {
                    const Vec3 axis = std::abs(normal.X) < .8 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
                    tangent = axis - normal * normal.Dot(axis);
                }
                tangent = tangent.Unit();
                const float handedness = normal.Cross(tangent).Dot(bitangentSums[i]) < 0 ? -1.f : 1.f;
                result.Vertices[i].Tangent = tangent.Direction(handedness);
            }
        }
    };
}

namespace resources {
    ObjData ObjLoader::Load(const std::filesystem::path& path) {
        const auto utf8 = path.u8string();
        const std::string source(utf8.begin(), utf8.end());
        std::ifstream input(path);
        if(!input) throw std::runtime_error("Cannot open OBJ: " + source);
        return Load(input, source);
    }

    ObjData ObjLoader::Load(std::istream& input, const std::string& sourceName) {
        return Parser(sourceName).Read(input);
    }
}
